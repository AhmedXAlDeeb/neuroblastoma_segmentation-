#!/usr/bin/env python3
"""
count_segmented_slices.py

Counts segmented slices in label NIfTI files and adds counts for unique files
found in another folder, ignoring duplicates (exact or highly similar).

Usage:
    python count_segmented_slices.py --labels /path/to/labels_dir --others /path/to/other_dir
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from typing import Tuple, Optional

# ---- Configurable thresholds ----
DUPLICATE_DICE_THRESHOLD = 0.9   # > => consider identical
SIMILAR_DICE_THRESHOLD = 0.9       # >= => consider similar (skip)
RESAMPLE_ORDER = 0                  # nearest for label masks (preserve labels)


def load_nifti_as_binary_mask(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NIfTI file and return (mask_bool, affine, header).
    mask_bool: ndarray of dtype bool where non-zero voxels are True.
    """
    img = nib.load(str(filepath))
    data = img.get_fdata(dtype=np.float32)  # float32 to avoid surprises, then threshold
    mask = data != 0
    return mask.astype(np.bool_), img.affine, img.header


def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dice coefficient for boolean arrays.
    If both are empty -> returns 1.0 (identical empties).
    """
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    a_sum = a_bool.sum()
    b_sum = b_bool.sum()
    if a_sum == 0 and b_sum == 0:
        return 1.0
    inter = np.logical_and(a_bool, b_bool).sum()
    denom = a_sum + b_sum
    if denom == 0:
        return 0.0
    return 2.0 * inter / denom


def resample_to_shape(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resample binary mask to target shape using nearest-neighbor (order=0).
    Uses scipy.ndimage.zoom with computed zoom factors.
    This is an approximate resampling that ignores affine information,
    but is robust if images share the same orientation and similar voxel spacing.
    """
    if mask.shape == tuple(target_shape):
        return mask
    zoom_factors = [t / s for s, t in zip(mask.shape, target_shape)]
    # use order=0 (nearest) to preserve labels
    res = ndi.zoom(mask.astype(np.float32), zoom_factors, order=RESAMPLE_ORDER)
    return (res > 0.5).astype(np.bool_)


def center_crop_or_pad_to_shape(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Fallback: center-crop or pad 'mask' to 'target_shape' if resampling is not desired.
    Keeps mask centered.
    """
    out = np.zeros(target_shape, dtype=np.bool_)
    src = mask
    src_shape = src.shape
    # compute starting indices to place src in out centered
    start = [max(0, (t - s) // 2) for s, t in zip(src_shape, target_shape)]
    # compute region sizes
    copy_size = [min(s, t) for s, t in zip(src_shape, target_shape)]
    # compute src starts
    src_start = [max(0, (s - t) // 2) for s, t in zip(src_shape, target_shape)]
    # slices
    out_slices = tuple(slice(start[i], start[i] + copy_size[i]) for i in range(3))
    src_slices = tuple(slice(src_start[i], src_start[i] + copy_size[i]) for i in range(3))
    out[out_slices] = src[src_slices]
    return out


def choose_slice_axis(mask: np.ndarray) -> int:
    """
    Heuristic to choose which axis corresponds to 'slices'. Usually the smallest dimension is the slice axis.
    Returns axis index.
    """
    shape = mask.shape
    axis = int(np.argmin(shape))
    return axis


def count_segmented_slices(mask: np.ndarray, slice_axis: Optional[int] = None) -> int:
    """
    Count number of 2D slices that contain any segmentation along the chosen slice_axis.
    If slice_axis is None, choose automatically (smallest axis).
    """
    if slice_axis is None:
        slice_axis = choose_slice_axis(mask)
    # axes to reduce for "any"
    reduce_axes = tuple(i for i in range(mask.ndim) if i != slice_axis)
    slices_any = np.any(mask, axis=reduce_axes)
    return int(slices_any.sum()), slice_axis


def find_nifti_files(folder: Path):
    exts = (".nii", ".nii.gz")
    return sorted([p for p in folder.iterdir() if p.suffix in exts or p.name.endswith(".nii.gz")])


def main(labels_dir: str, others_dir: str,
         duplicate_dice_thr: float = DUPLICATE_DICE_THRESHOLD,
         similar_dice_thr: float = SIMILAR_DICE_THRESHOLD,
         do_resample: bool = True):
    labels_dir = Path(labels_dir)
    others_dir = Path(others_dir)

    assert labels_dir.exists(), f"Labels folder not found: {labels_dir}"
    assert others_dir.exists(), f"Other folder not found: {others_dir}"

    label_files = find_nifti_files(labels_dir)
    other_files = find_nifti_files(others_dir)

    print(f"[INFO] Found {len(label_files)} files in labels folder and {len(other_files)} files in 'others' folder.\n")

    # Load all label masks first
    labels_info = []  # list of dicts: {'path': Path, 'mask': np.bool_, 'slices': int, 'nonzero': int, 'slice_axis': int}
    total_slices = 0

    print("=== Processing labels folder ===")
    for p in label_files:
        try:
            mask, affine, header = load_nifti_as_binary_mask(p)
        except Exception as e:
            print(f"[ERROR] Failed to load {p}: {e}")
            continue
        nonzero = int(mask.sum())
        slices_count, slice_axis = count_segmented_slices(mask)
        labels_info.append({'path': p, 'mask': mask, 'nonzero': nonzero, 'slices': slices_count, 'slice_axis': slice_axis})
        total_slices += slices_count
        print(f"[LABEL] {p.name}: shape={mask.shape}, nonzero_voxels={nonzero}, slice_axis={slice_axis}, segmented_slices={slices_count}")

    print(f"\n[INFO] Total segmented slices from labels folder: {total_slices}\n")

    # Process other folder, checking duplicates vs labels and within others
    unique_other_info = []  # accepted unique masks from others
    duplicates_skipped = 0
    added_slices_from_others = 0

    print("=== Processing 'others' folder ===")
    for p in other_files:
        print(f"\n[PROCESS] File: {p.name}")
        try:
            mask_o, affine_o, header_o = load_nifti_as_binary_mask(p)
        except Exception as e:
            print(f"[ERROR] Cannot load {p}: {e}")
            continue

        nonzero_o = int(mask_o.sum())
        if nonzero_o == 0:
            print("[SKIP] No segmentation present (all zeros).")
            continue

        # choose slice axis for this mask (used for counting slices later)
        slice_axis_o = choose_slice_axis(mask_o)
        print(f" - mask shape: {mask_o.shape}; nonzero_voxels: {nonzero_o}; chosen slice axis: {slice_axis_o}")

        # --- Check duplicates against labels first ---
        is_duplicate = False
        for li in labels_info:
            label_mask = li['mask']
            # try to align shapes: if shapes equal, compare directly; else resample or center-crop/pad
            if mask_o.shape == label_mask.shape:
                # exact equality quick check
                if np.array_equal(mask_o, label_mask):
                    print(f" - DUPLICATE (exact) with label: {li['path'].name}")
                    is_duplicate = True
                    break
                d = dice_coef(mask_o, label_mask)
                print(f"   dice vs label {li['path'].name}: {d:.6f}")
                if d >= duplicate_dice_thr:
                    print(f" - DUPLICATE (dice {d:.6f} >= {duplicate_dice_thr}) with label: {li['path'].name}")
                    is_duplicate = True
                    break
                if d >= similar_dice_thr:
                    print(f" - SIMILAR (dice {d:.6f} >= {similar_dice_thr}) to label: {li['path'].name} -> treat as duplicate")
                    is_duplicate = True
                    break
            else:
                # try resample (preferred) or fallback center-crop/pad
                aligned = None
                if do_resample:
                    try:
                        aligned = resample_to_shape(mask_o, label_mask.shape)
                        d = dice_coef(aligned, label_mask)
                        print(f"   resampled dice vs label {li['path'].name}: {d:.6f} (resampled {mask_o.shape} -> {label_mask.shape})")
                        if d >= duplicate_dice_thr:
                            print(f" - DUPLICATE (resampled dice {d:.6f} >= {duplicate_dice_thr}) with label: {li['path'].name}")
                            is_duplicate = True
                            break
                        if d >= similar_dice_thr:
                            print(f" - SIMILAR (resampled dice {d:.6f} >= {similar_dice_thr}) to label: {li['path'].name} -> treat as duplicate")
                            is_duplicate = True
                            break
                    except Exception as e:
                        print(f"   [WARN] resampling failed: {e}. Trying center-crop/pad fallback.")
                        aligned = center_crop_or_pad_to_shape(mask_o, label_mask.shape)
                        d = dice_coef(aligned, label_mask)
                        print(f"   fallback dice vs label {li['path'].name}: {d:.6f}")
                        if d >= similar_dice_thr:
                            print(f" - SIMILAR (fallback dice {d:.6f} >= {similar_dice_thr}) to label: {li['path'].name} -> treat as duplicate")
                            is_duplicate = True
                            break

        if is_duplicate:
            duplicates_skipped += 1
            continue

        # --- Check duplicates against previously accepted OTHER uniques ---
        for ui in unique_other_info:
            target_mask = ui['mask']
            if mask_o.shape == target_mask.shape:
                if np.array_equal(mask_o, target_mask):
                    print(f" - DUPLICATE (exact) with earlier other-file: {ui['path'].name}")
                    is_duplicate = True
                    break
                d = dice_coef(mask_o, target_mask)
                print(f"   dice vs other {ui['path'].name}: {d:.6f}")
                if d >= duplicate_dice_thr:
                    print(f" - DUPLICATE (dice {d:.6f} >= {duplicate_dice_thr}) with other: {ui['path'].name}")
                    is_duplicate = True
                    break
                if d >= similar_dice_thr:
                    print(f" - SIMILAR (dice {d:.6f} >= {similar_dice_thr}) to other: {ui['path'].name} -> treat as duplicate")
                    is_duplicate = True
                    break
            else:
                # try to resample to target shape
                aligned = None
                try:
                    aligned = resample_to_shape(mask_o, target_mask.shape)
                    d = dice_coef(aligned, target_mask)
                    print(f"   resampled dice vs other {ui['path'].name}: {d:.6f}")
                    if d >= similar_dice_thr:
                        print(f" - SIMILAR (resampled dice {d:.6f} >= {similar_dice_thr}) to other: {ui['path'].name} -> treat as duplicate")
                        is_duplicate = True
                        break
                except Exception as e:
                    print(f"   [WARN] resample vs other failed: {e}. Trying center-crop/pad fallback.")
                    aligned = center_crop_or_pad_to_shape(mask_o, target_mask.shape)
                    d = dice_coef(aligned, target_mask)
                    print(f"   fallback dice vs other {ui['path'].name}: {d:.6f}")
                    if d >= similar_dice_thr:
                        print(f" - SIMILAR (fallback dice {d:.6f} >= {similar_dice_thr}) to other: {ui['path'].name} -> treat as duplicate")
                        is_duplicate = True
                        break

        if is_duplicate:
            duplicates_skipped += 1
            continue

        # If we reached here -> not duplicate, accept and count segmented slices
        slices_count, _ = count_segmented_slices(mask_o, slice_axis=slice_axis_o)
        total_slices += slices_count
        added_slices_from_others += slices_count
        unique_other_info.append({'path': p, 'mask': mask_o, 'nonzero': nonzero_o, 'slices': slices_count})
        print(f"[ACCEPT] {p.name}: segmented_slices={slices_count} -> ADDED to total.")

    # Final report
    print("\n=== SUMMARY ===")
    print(f"Labels files processed : {len(labels_info)}")
    print(f"Other files processed  : {len(other_files)}")
    print(f"Other duplicates skipped: {duplicates_skipped}")
    print(f"Unique other accepted  : {len(unique_other_info)}")
    print(f"Slices added from others: {added_slices_from_others}")
    print(f"TOTAL segmented slices (labels + unique others): {total_slices}")


if __name__ == "__main__":
    others_dir = "D:/SBME/GP/Neuroblastoma/neuroblastoma_segmentation/data/more_segmentations"
    
    main("D:/SBME/GP/Neuroblastoma/neuroblastoma_segmentation/data/nnUNet_raw/Dataset001_Neuroblastoma/labelsTr", others_dir, duplicate_dice_thr=DUPLICATE_DICE_THRESHOLD, similar_dice_thr=SIMILAR_DICE_THRESHOLD, do_resample=True)
