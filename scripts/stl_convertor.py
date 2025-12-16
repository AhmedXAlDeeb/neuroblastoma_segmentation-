#!/usr/bin/env python3
"""
STL -> NIfTI converter (robust rasterization & alignment)

Approach:
1. Load mesh with trimesh
2. Try simple repairs if mesh is not watertight
3. Load reference NIfTI and compute world coordinates of voxel centers using affine
4. Test voxel centers for inclusion inside mesh (mesh.contains), in chunks
5. Build binary mask and save as NIfTI with the same affine as reference

This produces a filled mask aligned to the reference image.
"""

import os
import sys
import traceback
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import trimesh

import vtk


def log(msg: str):
    print(msg)
    sys.stdout.flush()


def try_repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attempt light-weight repairs to increase chance of contains() working."""
    try:
        # Ensure correct winding / normals
        if not mesh.is_watertight:
            log("  ⚠ mesh not watertight: attempting repairs (fill_holes/fix_normals)...")
            try:
                trimesh.repair.fill_holes(mesh)
            except Exception as e:
                log(f"    fill_holes failed: {e}")

            try:
                trimesh.repair.fix_normals(mesh)
            except Exception as e:
                log(f"    fix_normals failed: {e}")

            # Merge duplicate vertices / remove degenerate faces
            try:
                mesh.remove_unreferenced_vertices()
                mesh.remove_degenerate_faces()
            except Exception:
                pass

        return mesh
    except Exception as e:
        log(f"  ⚠ Mesh repair threw: {e}")
        return mesh


def voxel_centers_world(shape: Tuple[int, int, int], affine: np.ndarray, chunk: slice = None):
    """
    Generator that yields (indices_slice, centers_world) where centers_world is (N,3)
    for the requested sub-block of voxels.
    - shape: (nx,ny,nz)
    - affine: 4x4 affine mapping voxel indices to world (voxel i-> world: affine @ [i+0.5,...,1])
    - chunk: slice object for flatten index range (optional)
    """
    nx, ny, nz = shape
    # flatten count
    total = nx * ny * nz

    if chunk is None:
        start = 0
        stop = total
    else:
        start = int(chunk.start)
        stop = int(chunk.stop)

    # We'll compute indices for the given flattened range
    # convert flattened k in [start, stop) to i,j,k using division
    count = stop - start
    if count <= 0:
        return

    # generate linear indices in blocks to avoid giant arrays
    # produce grid of linear indices
    idx = np.arange(start, stop, dtype=np.int64)
    i = idx // (ny * nz)
    rem = idx % (ny * nz)
    j = rem // nz
    k = rem % nz

    # convert to world coords of voxel centers (i+0.5, j+0.5, k+0.5)
    ijk = np.vstack((i + 0.5, j + 0.5, k + 0.5, np.ones_like(i, dtype=float)))
    # world = affine @ ijk
    world = affine @ ijk
    world3 = world[:3, :].T  # (N,3)
    return (start, stop), world3, (i, j, k)


def mesh_contains_mask(mesh: trimesh.Trimesh, ref_shape: Tuple[int, int, int], ref_affine: np.ndarray, chunk_size: int = 2_000_000):
    """
    Create boolean mask where True = inside mesh.
    We test voxel centers in chunks of chunk_size (flattened voxels).
    """
    nx, ny, nz = ref_shape
    total = nx * ny * nz
    mask = np.zeros(total, dtype=np.uint8)

    # loop chunks
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        (s, e), world3, (ii, jj, kk) = voxel_centers_world(ref_shape, ref_affine, chunk=slice(start, stop))
        # trimesh.contains expects (N,3)
        try:
            inside = mesh.contains(world3)
        except Exception as ex:
            # If contains fails (e.g. mesh non-watertight), try repairing and retry once
            log(f"  ⚠ mesh.contains error: {ex} — attempting repair + retry")
            mesh = try_repair_mesh(mesh)
            inside = mesh.contains(world3)

        # inside is boolean array
        mask[s:e] = inside.astype(np.uint8)

        log(f"    processed voxels {s}..{e} -> inside count: {inside.sum()}")

    mask3 = mask.reshape((nx, ny, nz))
    return mask3


def mesh_to_nifti_fill(stl_path: str, output_dir: str, reference_nifti_path: str):
    """
    Robust STL → NIfTI converter using VTK stencil rasterization.
    No raycasting; handles large meshes efficiently.
    """
    print(f"\n🧱 Converting {stl_path}")
    
    # --- Load reference NIfTI ---
    ref_img = nib.load(reference_nifti_path)
    affine = ref_img.affine
    spacing = np.abs(np.diag(affine)[:3])
    shape = ref_img.shape[:3]
    origin = affine[:3, 3]
    
    # --- Load STL ---
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    poly = reader.GetOutput()
    
    # --- Create blank image matching reference ---
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(shape)
    image.SetOrigin(origin)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)
    
    # --- Rasterize STL into binary stencil ---
    poly2stencil = vtk.vtkPolyDataToImageStencil()
    poly2stencil.SetInputData(poly)
    poly2stencil.SetOutputSpacing(spacing)
    poly2stencil.SetOutputOrigin(origin)
    poly2stencil.SetOutputWholeExtent(image.GetExtent())
    poly2stencil.Update()
    
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(poly2stencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()
    
    # --- Convert to NumPy array ---
    vtk_array = stencil.GetOutput().GetPointData().GetScalars()
    mask_np = np.flipud(np.transpose(np.reshape(
        np.frombuffer(vtk_array, dtype=np.uint8),
        shape[::-1]  # vtk stores in z,y,x order
    )))
    
    # --- Save NIfTI ---
    out_path = Path(output_dir) / (Path(stl_path).stem + ".nii.gz")
    nib.save(nib.Nifti1Image(mask_np.astype(np.uint8), affine), str(out_path))
    print(f"✅ Saved mask: {out_path}")
    return str(out_path)
def process_directory(root_dir: str, chunk_size: int = 2_000_000):
    log(f"🚀 Starting batch STL → NIfTI in: {root_dir}")
    total_success = 0
    total_failed = 0
    for case in sorted(os.listdir(root_dir)):
        case_path = os.path.join(root_dir, case)
        if not os.path.isdir(case_path):
            continue
        stl_files = [f for f in os.listdir(case_path) if f.lower().endswith(".stl")]
        nii_files = [f for f in os.listdir(case_path) if f.lower().endswith((".nii", ".nii.gz"))]
        if not stl_files or not nii_files:
            log(f"⚠ Skipping {case} (missing STL or reference NIfTI)")
            continue
        ref_img = os.path.join(case_path, nii_files[0])
        for stl_file in stl_files:
            stl_path = os.path.join(case_path, stl_file)
            try:
                mesh_to_nifti_fill(stl_path, case_path, ref_img)
                total_success += 1
            except Exception as e:
                log(f"  ❌ Error converting {stl_file}: {e}")
                log(traceback.format_exc())
                total_failed += 1
    log("\nSummary:")
    log(f"  Successful: {total_success}")
    log(f"  Failed: {total_failed}")
    log(f"  Total processed: {total_success + total_failed}")


if __name__ == "__main__":
    # Example usage: python stl_to_nifti.py /path/to/stl_root
    
    root = r"D:\SBME\GP\Neuroblastoma\neuroblastoma_segmentation\data\stl_data"
    process_directory(root)
