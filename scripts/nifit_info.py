import os
import nibabel as nib
import numpy as np

def print_nifti_info(nifti_path: str):
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    print(f"📂 File: {nifti_path}")
    print("=" * 80)

    # Load NIfTI
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    hdr = img.header
    affine = img.affine

    # --- Basic Info ---
    print("🧾 Basic Information")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {hdr.get_data_dtype()}")
    print(f"  Memory size: {data.nbytes / (1024 ** 2):.2f} MB")
    print()

    # --- Spatial Info ---
    print("🌍 Spatial Information")
    print(f"  Voxel dimensions (spacing): {hdr.get_zooms()}")
    print(f"  Affine matrix:\n{affine}")
    print(f"  Origin (from affine): {affine[:3, 3]}")
    print()

    # --- Intensity Statistics ---
    print("📊 Intensity Statistics")
    print(f"  Min intensity: {np.min(data):.4f}")
    print(f"  Max intensity: {np.max(data):.4f}")
    print(f"  Mean intensity: {np.mean(data):.4f}")
    print(f"  Std intensity: {np.std(data):.4f}")
    print(f"  Non-zero voxels: {np.count_nonzero(data)} / {data.size}")
    print()

    # --- Orientation Info ---
    print("🧭 Orientation Information")
    orientation = nib.aff2axcodes(affine)
    print(f"  Anatomical orientation: {orientation}")
    print()

    # --- Header Summary ---
    print("🧩 Header Fields")
    for key in ["datatype", "bitpix", "sform_code", "qform_code", "xyzt_units"]:
        if key in hdr:
            print(f"  {key}: {hdr[key]}")
    print()

    # --- Sanity Checks ---
    print("✅ Sanity Checks")
    if np.count_nonzero(data) == 0:
        print("  ⚠️  WARNING: All voxels are zero — possible empty mask or wrong alignment.")
    if np.isnan(data).any():
        print("  ⚠️  WARNING: NaN values found in the image.")
    if np.isinf(data).any():
        print("  ⚠️  WARNING: Infinite values found in the image.")

    print("=" * 80)
    print("✔️ Done.\n")



if __name__ == "__main__":
    # 🧠 Example usage
    nifti_path = r"D:\SBME\GP\Neuroblastoma\neuroblastoma_segmentation\data\stl_data\case_20\case 20 portovenous phase.nii.gz"
    print_nifti_info(nifti_path)
