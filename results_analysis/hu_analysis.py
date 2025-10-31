import os
import nibabel as nib
import numpy as np
import pandas as pd

def get_hu_stats(ct_data, mask, name):
    """Compute HU statistics for a given binary mask."""
    if np.sum(mask) == 0:
        return {f'{name}_mean_hu': np.nan, f'{name}_std_hu': np.nan,
                f'{name}_min_hu': np.nan, f'{name}_max_hu': np.nan,
                f'{name}_median_hu': np.nan}
    hu_values = ct_data[mask > 0]
    return {
        f'{name}_mean_hu': float(np.mean(hu_values)),
        f'{name}_std_hu': float(np.std(hu_values)),
        f'{name}_min_hu': float(np.min(hu_values)),
        f'{name}_max_hu': float(np.max(hu_values)),
        f'{name}_median_hu': float(np.median(hu_values))
    }

def analyze_hu_only(scan_path, label_path, case_name):
    """Perform HU analysis for a single case (manual only)."""
    ct_data = nib.load(scan_path).get_fdata()
    label = (nib.load(label_path).get_fdata() > 0).astype(np.uint8)

    results = {'case': case_name}
    results.update(get_hu_stats(ct_data, label, 'manual'))
    return results

def batch_hu_analysis(fold_num, base_dir="."):
    """Run HU analysis for all cases in one fold (manual only)."""
    print(f"🧩 Processing fold {fold_num}...")

    scan_dir = os.path.join(base_dir, f"fold_{fold_num}", "scans")
    label_dir = os.path.join(base_dir, f"fold_{fold_num}", "labels")
    results_dir = os.path.join(base_dir, f"fold_{fold_num}", "results", "hu_analysis")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for scan_file in sorted(os.listdir(scan_dir)):
        if not scan_file.endswith(".nii.gz"):
            continue

        case_num = scan_file.replace("case_", "").replace(".nii.gz", "")
        label_file = f"case_0{case_num}.nii.gz"

        scan_path = os.path.join(scan_dir, scan_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"⚠ Label missing for case {case_num}, skipping.")
            continue

        try:
            result = analyze_hu_only(scan_path, label_path, case_num)
            all_results.append(result)
            print(f"  ✅ Case {case_num} processed.")
        except Exception as e:
            print(f"  ❌ Error in case {case_num}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(results_dir, "hu_statistics.csv")
        df.to_csv(out_path, index=False)
        print(f"📊 Saved HU statistics to: {out_path}")
    else:
        print("❌ No valid cases processed.")

if __name__ == "__main__":
    scan_dir = r"D:\SBME\GP\Neuroblastoma\neuroblastoma_segmentation\data\nnUNet_raw\Dataset001_Neuroblastoma\imagesTr"    # update with your scans directory
    label_dir = r"D:\SBME\GP\Neuroblastoma\neuroblastoma_segmentation\data\nnUNet_raw\Dataset001_Neuroblastoma\labelsTr"  # update with your labels directory
    results_dir = "./results/hu_analysis"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for scan_file in sorted(os.listdir(scan_dir)):
        if not scan_file.endswith(".nii.gz"):
            continue

        # Extract case number from scan filename (e.g., case_001_0000.nii.gz -> 001)
        base_name = scan_file.split(".nii.gz")[0]
        case_num = base_name.split("_")[1]  # '001' from 'case_001_0000'
        label_file = f"case_{case_num}.nii.gz"
        scan_path = os.path.join(scan_dir, scan_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"⚠ Label missing for case {case_num}, skipping.")
            continue

        try:
            result = analyze_hu_only(scan_path, label_path, case_num)
            all_results.append(result)
            print(f"  ✅ Case {case_num} processed.")
        except Exception as e:
            print(f"  ❌ Error in case {case_num}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(results_dir, "hu_statistics.csv")
        df.to_csv(out_path, index=False)
        print(f"📊 Saved HU statistics to: {out_path}")
    else:
        print("❌ No valid cases processed.")
