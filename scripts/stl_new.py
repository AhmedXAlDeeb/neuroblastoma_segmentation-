#!/usr/bin/env python3
"""
STL -> NIfTI converter with automatic transformation to fit the reference volume.

This version:
✅ Loads STL mesh
✅ Automatically scales and centers it to fit the reference NIfTI volume
✅ Rasterizes it to binary mask using VTK
✅ Saves the aligned NIfTI mask in the same folder

Author: Adapted for Ahmed Soliman
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import vtk


def log(msg: str):
    print(msg)
    sys.stdout.flush()


def mesh_to_nifti_fill(stl_path: str, output_dir: str, reference_nifti_path: str):
    """
    STL → NIfTI converter with automatic alignment.
    The STL is scaled and centered to match the reference NIfTI volume.
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

    # --- Auto-fit STL to NIfTI volume ---
    bounds = np.array(poly.GetBounds())  # (xmin, xmax, ymin, ymax, zmin, zmax)
    stl_size = bounds[1::2] - bounds[::2]
    ref_size = np.array(shape) * spacing

    # Scale STL to fit slightly smaller than reference volume
    scale_factor = np.min(ref_size / stl_size) * 0.9
    transform = vtk.vtkTransform()
    transform.Scale(scale_factor, scale_factor, scale_factor)

    # Compute centers for translation
    stl_center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    ])
    ref_center = origin + ref_size / 2
    translation = ref_center - stl_center * scale_factor
    transform.Translate(translation)

    # Apply the transformation
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(poly)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    poly = transform_filter.GetOutput()

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


def process_directory(root_dir: str):
    """
    Batch conversion for all cases under root_dir.
    Each case folder must contain one STL and one NIfTI file.
    """
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
