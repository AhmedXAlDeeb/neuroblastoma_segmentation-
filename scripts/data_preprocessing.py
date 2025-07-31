import os
import SimpleITK as sitk

# ==== Set your base directory here ====
base_path = r"D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-\data\raw\Dataset001_Neuroblastoma"
images_dir = os.path.join(base_path, "imagesTr")
labels_dir = os.path.join(base_path, "labelsTr")
output_labels_dir = os.path.join(base_path, "labelsTr_fixed")

os.makedirs(output_labels_dir, exist_ok=True)

# ==== Helper: resample label to image space ====
def resample_label_to_image(image_path, label_path, output_path):
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Check shape/spacing
    if image.GetSize() != label.GetSize() or image.GetSpacing() != label.GetSpacing() \
        or image.GetOrigin() != label.GetOrigin() or image.GetDirection() != label.GetDirection():
        print(f"⚠ Resampling: {os.path.basename(label_path)}")

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(image)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetTransform(sitk.Transform())
        resampled_label = resample.Execute(label)

        sitk.WriteImage(resampled_label, output_path)
    else:
        print(f"✅ No changes needed: {os.path.basename(label_path)}")
        sitk.WriteImage(label, output_path)

# ==== Process all cases ====
for fname in os.listdir(images_dir):
    if not fname.endswith("_0000.nii.gz"):
        continue

    case_id = fname.replace("_0000.nii.gz", "")
    image_path = os.path.join(images_dir, fname)
    label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
    output_path = os.path.join(output_labels_dir, f"{case_id}.nii.gz")

    if not os.path.exists(label_path):
        print(f"❌ Missing label for {case_id}")
        continue

    try:
        resample_label_to_image(image_path, label_path, output_path)
    except Exception as e:
        print(f"❌ Error processing {case_id}: {e}")

print("\n✅ Done fixing all label files!")
print(f"🔁 Fixed labels saved to: {output_labels_dir}")
