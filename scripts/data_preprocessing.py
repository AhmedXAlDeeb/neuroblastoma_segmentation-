import os
import SimpleITK as sitk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==== Set your base directory here ====
base_path = os.getenv("DATA_DIR")  # Make sure DATA_DIR is set in your .env
images_dir = os.path.join(base_path, "imagesTr")
labels_dir = os.path.join(base_path, "labelsTr")
labels_old_dir = os.path.join(base_path, "labelsTr_old")
labels_fixed_dir = os.path.join(base_path, "labelsTr_fixed")

os.makedirs(labels_fixed_dir, exist_ok=True)

# ==== Step 1: Resample labels to match images ====
def resample_label_to_image(image_path, label_path, output_path):
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Force resample label into image space
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  # preserve labels
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)  # background

    resampled_label = resample.Execute(label)

    # Ensure uint8 for segmentation
    resampled_label = sitk.Cast(resampled_label, sitk.sitkUInt8)
    sitk.WriteImage(resampled_label, output_path)
    print(f"✔ Fixed {os.path.basename(label_path)}")

# Process all labels
for fname in os.listdir(labels_dir):
    if not fname.endswith(".nii.gz"):
        continue
    case_id = fname.replace(".nii.gz", "")
    image_path = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
    label_path = os.path.join(labels_dir, fname)
    output_path = os.path.join(labels_fixed_dir, fname)

    if not os.path.exists(image_path):
        print(f"⚠ No image found for {fname}, skipping")
        continue

    resample_label_to_image(image_path, label_path, output_path)

# ==== Step 2: Rename folders (backup old, replace with new) ====
if os.path.exists(labels_dir) and not os.path.exists(labels_old_dir):
    os.rename(labels_dir, labels_old_dir)
    print(f"✅ Renamed {labels_dir} → {labels_old_dir}")
else:
    print("⚠ labelsTr already renamed or missing")

if os.path.exists(labels_fixed_dir):
    os.rename(labels_fixed_dir, labels_dir)
    print(f"✅ Renamed {labels_fixed_dir} → {labels_dir}")
else:
    print("⚠ labelsTr_fixed not found")

# ==== Step 3: Verify shapes and spacing ====
print("\n🔎 Verifying shapes and spacings after correction...\n")
for fname in os.listdir(labels_dir):
    if not fname.endswith(".nii.gz"):
        continue
    case_id = fname.replace(".nii.gz", "")
    image_path = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
    label_path = os.path.join(labels_dir, fname)

    if not os.path.exists(image_path):
        continue

    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    if image.GetSize() != label.GetSize():
        print(f"❌ Shape mismatch: {fname}")
        print(f"   Image: {image.GetSize()} | Label: {label.GetSize()}")
    elif image.GetSpacing() != label.GetSpacing():
        print(f"❌ Spacing mismatch: {fname}")
        print(f"   Image: {image.GetSpacing()} | Label: {label.GetSpacing()}")
    elif image.GetOrigin() != label.GetOrigin():
        print(f"⚠ Origin mismatch: {fname}")
        print(f"   Image: {image.GetOrigin()} | Label: {label.GetOrigin()}")
    elif image.GetDirection() != label.GetDirection():
        print(f"⚠ Direction mismatch: {fname}")
    else:
        print(f"✅ {fname} matches perfectly")
