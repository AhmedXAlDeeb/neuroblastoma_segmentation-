import os
import SimpleITK as sitk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==== Set your base directory here ====
base_path = os.getenv("DATA_DIR")  # Make sure DATA_DIR is set in your .env
images_dir = os.path.join(base_path, "imagesTr")
labels_dir = os.path.join(base_path, "labelsTr")
output_labels_dir = os.path.join(base_path, "labelsTr_fixed")

os.makedirs(output_labels_dir, exist_ok=True)

# ==== Helper: resample label to image space ====
def resample_label_to_image(image_path, label_path, output_path):
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Check if metadata matches
    needs_resample = (
        image.GetSize() != label.GetSize()
        or image.GetSpacing() != label.GetSpacing()
        or image.GetOrigin() != label.GetOrigin()
        or image.GetDirection() != label.GetDirection()
    )

    if needs_resample:
        print(f"⚠ Resampling: {os.path.basename(label_path)}")

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(image)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)  # keep labels discrete
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)  # background
        resampled_label = resample.Execute(label)

        # Ensure integer type
        resampled_label = sitk.Cast(resampled_label, sitk.sitkInt16)
        sitk.WriteImage(resampled_label, output_path)
    else:
        print(f"✅ No changes needed: {os.path.basename(label_path)}")
        # Still cast to int16
