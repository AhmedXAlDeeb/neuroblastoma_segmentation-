import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff
from medpy.metric import binary

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2. * intersection / (y_true.sum() + y_pred.sum() + 1e-8)

def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-8)

def precision_score(y_true, y_pred):
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    return tp / (tp + fp + 1e-8)

def recall_score(y_true, y_pred):
    tp = np.logical_and(y_true, y_pred).sum()
    fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()
    return tp / (tp + fn + 1e-8)

def hausdorff95(y_true, y_pred):
    try:
        return binary.hd95(y_pred, y_true)
    except:
        return np.nan

def avg_surface_distance(y_true, y_pred):
    try:
        return binary.asd(y_pred, y_true)
    except:
        return np.nan

def evaluate_segmentation(pred_path, label_path):
    pred = nib.load(pred_path).get_fdata() > 0
    lab = nib.load(label_path).get_fdata() > 0

    return {
        "dice": dice_score(lab, pred),
        "jaccard": jaccard_index(lab, pred),
        "precision": precision_score(lab, pred),
        "recall": recall_score(lab, pred),
        "hausdorff95": hausdorff95(lab, pred),
        "asd": avg_surface_distance(lab, pred),
    }

def batch_evaluate_all_folds(num_folds=4):
    results = {}
    for fold_num in range(num_folds):
        print(f"Evaluating fold {fold_num}...")
        fold_dir = f"fold_{fold_num}"
        post_dir = os.path.join(fold_dir, "postprocessed_pred")
        label_dir = os.path.join(fold_dir, "labels")

        fold_metrics = []
        pred_cases = [f for f in os.listdir(post_dir) if f.endswith('.nii.gz')]
        for pred_file in pred_cases:
            print(f"  Evaluating {pred_file}...")
            label_file = pred_file
            pred_path = os.path.join(post_dir, pred_file)
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                continue

            metrics = evaluate_segmentation(pred_path, label_path)
            fold_metrics.append(metrics)

        results[f"fold_{fold_num}"] = fold_metrics

    return results


if __name__ == "__main__":
    # 1. Make sure postprocessing is already done!
    # batch_postprocess_all_folds(num_folds=4)  # <-- Remove or comment out this line

    # 2. Evaluate metrics on postprocessed segmentations
    metrics = batch_evaluate_all_folds(num_folds=4)
    import json
    with open("segmentation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved results to segmentation_metrics.json")
