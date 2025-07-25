import os
import nibabel as nib
from pathlib import Path

# Path to raw data folder relative to the script
root_dir = Path(__file__).resolve().parent.parent / "raw_data"

# Loop through case 1 to case 31
for i in range(1, 32):
    case_dir = root_dir / f"case {i}"

    if not case_dir.exists():
        print(f"Skipping missing folder: {case_dir}")
        continue

    print(f"Processing folder: {case_dir}")

    # --- Rename PNG files ---
    png_files = sorted(case_dir.glob("*.png"))
    for idx, file in enumerate(png_files):
        new_name = f"screenshot_{idx:03}.png"
        new_path = case_dir / new_name
        file.rename(new_path)
        print(f"Renamed {file.name} → {new_name}")

    # --- Convert .nii to .nii.gz ---
    nii_files = case_dir.glob("*.nii")
    for nii_file in nii_files:
        try:
            img = nib.load(str(nii_file))
            gz_path = nii_file.with_suffix(".nii.gz")
            nib.save(img, str(gz_path))
            print(f"Converted {nii_file.name} → {gz_path.name}")
            nii_file.unlink()  # Delete original .nii
        except Exception as e:
            print(f"❌ Failed to convert {nii_file.name}: {e}")
