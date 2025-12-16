import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label

def postprocess_keep_common_volumes(pred_path, label_path, out_path):
    """
    Keep only predicted volumes that overlap with at least one label volume.
    """
    pred_img = nib.load(pred_path)
    label_img = nib.load(label_path)

    pred = (pred_img.get_fdata() > 0).astype(np.uint8)
    lab = (label_img.get_fdata() > 0).astype(np.uint8)

    # Connected components in pred and label
    pred_cc, n_pred = label(pred)
    lab_cc, n_lab = label(lab)

    keep_mask = np.zeros_like(pred, dtype=np.uint8)

    # For each predicted component, check if it overlaps with any label component
    for i in range(1, n_pred + 1):
        pred_vol = (pred_cc == i)
        overlap = np.logical_and(pred_vol, lab > 0)
        if overlap.any():  # keep the whole predicted volume
            keep_mask[pred_vol] = 1

    # Save new NIfTI
    keep_img = nib.Nifti1Image(keep_mask, affine=pred_img.affine, header=pred_img.header)
    nib.save(keep_img, out_path)
    print(f"Saved common volumes mask: {out_path}")

def batch_postprocess_all_folds(num_folds=4):
    for fold_num in range(num_folds):
        fold_dir = f"fold_{fold_num}"
        pred_dir = os.path.join(fold_dir, "pred")
        label_dir = os.path.join(fold_dir, "labels")
        out_dir = os.path.join(fold_dir, "postprocessed_pred")
        os.makedirs(out_dir, exist_ok=True)

        pred_cases = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
        for pred_file in pred_cases:
            label_file = pred_file  # assumes same naming
            pred_path = os.path.join(pred_dir, pred_file)
            label_path = os.path.join(label_dir, label_file)
            out_path = os.path.join(out_dir, pred_file)

            if not os.path.exists(label_path):
                print(f"Label file missing for {pred_file}, skipping.")
                continue

            postprocess_keep_common_volumes(pred_path, label_path, out_path)

if __name__ == "__main__":
    batch_postprocess_all_folds(num_folds=4)  # Change num_folds if needed
