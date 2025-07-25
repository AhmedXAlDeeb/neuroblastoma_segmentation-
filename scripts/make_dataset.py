import os
import shutil
import random
from pathlib import Path

# ---------- CONFIGURATION ----------
src_root = Path(__file__).resolve().parent.parent / "raw_data"
dst_root = Path(__file__).resolve().parent.parent / "data/raw/Dataset001_Neuroblastoma"
imagesTr_dir = dst_root / "imagesTr"
labelsTr_dir = dst_root / "labelsTr"

# Create destination dirs
imagesTr_dir.mkdir(parents=True, exist_ok=True)
labelsTr_dir.mkdir(parents=True, exist_ok=True)

valid_cases = []

# ---------- RENAME AND COLLECT FILES ----------
for i in range(1, 32):
    case_id = f"{i:02}"
    case_dir = src_root / f"case {i}"

    if not case_dir.exists():
        print(f"⏭️ Skipping missing folder: {case_dir}")
        continue

    label_dst = case_dir / f"label_{case_id}.nii.gz"
    image_dst = case_dir / f"image_{case_id}.nii.gz"

    # Skip if files already exist and are valid
    if label_dst.exists() and image_dst.exists():
        valid_cases.append((case_id, image_dst, label_dst))
        continue

    label_file = None
    image_file = None

    for f in case_dir.glob("*.nii*"):
        fname = f.name.lower()
        if "segment" in fname and not label_dst.exists():
            label_file = f
        elif "segment" not in fname and not image_dst.exists():
            image_file = f

    try:
        if label_file and not label_dst.exists():
            label_file.rename(label_dst)
        if image_file and not image_dst.exists():
            image_file.rename(image_dst)
    except Exception as e:
        print(f"❌ Rename failed in case {case_id}: {e}")
        continue

    if label_dst.exists() and image_dst.exists():
        valid_cases.append((case_id, image_dst, label_dst))
    else:
        print(f"⚠️ Skipping case {case_id}: missing image or label")

# ---------- SPLIT ----------
random.seed(42)
random.shuffle(valid_cases)
train_split = int(len(valid_cases) * 0.8)
train_cases = valid_cases[:train_split]
test_cases = valid_cases[train_split:]

# ---------- COPY TO nnUNet FORMAT ----------
for idx, (case_id, img_path, lbl_path) in enumerate(train_cases):
    case_name = f"case_{idx:04}"
    dst_img = imagesTr_dir / f"{case_name}_0000.nii.gz"
    dst_lbl = labelsTr_dir / f"{case_name}.nii.gz"

    try:
        shutil.copy(img_path, dst_img)
        shutil.copy(lbl_path, dst_lbl)
        print(f"✅ Copied: {img_path.name} and {lbl_path.name} → {case_name}")
    except Exception as e:
        print(f"❌ Failed to copy {case_name}: {e}")

print(f"\n📦 Total training cases copied: {len(train_cases)}")
print(f"⛔ Skipped/invalid cases: {31 - len(valid_cases)}")
