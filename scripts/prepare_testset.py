import os
import shutil
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.paths import nnUNet_raw

def prepare_test_images(dataset_id: str, source_folder: str):
    """
    Validates, renames, and moves test images into nnUNet_raw/DatasetXXX/imagesTs
    following nnU-Net naming conventions.
    """
    # ---------------------------------
    # 1. Define Paths
    # ---------------------------------
    target_folder =r"C:\Neuroblastoma\Neuroblastoma\neuroblastoma_segmentation\data\nnUNet_raw\Dataset001_Neuroblastoma\imagesTs"
    maybe_mkdir_p(target_folder)

    # ---------------------------------
    # 2. Get all nii.gz files
    # ---------------------------------
    nii_files = [f for f in os.listdir(source_folder) if f.endswith(".nii.gz")]
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {source_folder}")

    # ---------------------------------
    # 3. Validate and Rename
    # ---------------------------------
    for i, file_name in enumerate(sorted(nii_files)):
        src_path = join(source_folder, file_name)

        # Try loading with nibabel to ensure it's a valid NIfTI file
        try:
            nib.load(src_path)
        except Exception as e:
            print(f"Skipping invalid file {file_name}: {e}")
            continue

        # Extract case name without extension
        case_name = os.path.splitext(os.path.splitext(file_name)[0])[0]

        # Ensure correct naming convention: case_0000.nii.gz
        new_name = f"{case_name}_0000.nii.gz"
        dst_path = join(target_folder, new_name)

        # Copy and rename
        shutil.copy(src_path, dst_path)
        print(f"Copied and renamed: {file_name} → {new_name}")

    print(f"All valid test images moved to: {target_folder}")


if __name__ == "__main__":
    # ---------------------------------
    # Example Usage
    # ---------------------------------
    dataset_id = "Dataset001_Neuroblastoma"  # change as needed
    source_folder = r"C:\Neuroblastoma\Neuroblastoma\neuroblastoma_segmentation\data\test\images"  # folder containing raw test images
    prepare_test_images(dataset_id, source_folder)
