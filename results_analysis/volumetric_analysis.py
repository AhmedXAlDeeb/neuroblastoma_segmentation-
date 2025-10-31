import nibabel as nib
import numpy as np
import pandas as pd
from skimage import measure
import os

# -------------------
# 1. Basic volume
# -------------------
def calculate_volume(mask_path):
    nii = nib.load(mask_path)
    mask = nii.get_fdata() > 0

    voxel_spacing = nii.header.get_zooms()  # (x, y, z) in mm
    voxel_volume = np.prod(voxel_spacing)   # mm^3 per voxel

    volume_mm3 = mask.sum() * voxel_volume
    volume_cm3 = volume_mm3 / 1000
    print(f"Volume (mm³): {volume_mm3}, Volume (cm³): {volume_cm3}")
    return volume_mm3, volume_cm3, voxel_spacing


# -------------------
# 2. Region-based volumes (if mask has multiple labels)
# -------------------
def region_volumes(mask_path):
    nii = nib.load(mask_path)
    mask = nii.get_fdata().astype(int)

    voxel_spacing = nii.header.get_zooms()
    voxel_volume = np.prod(voxel_spacing)

    results = {}
    for region_id in np.unique(mask):
        if region_id == 0:  # skip background
            continue
        region_volume_mm3 = np.sum(mask == region_id) * voxel_volume
        region_volume_cm3 = region_volume_mm3 / 1000
        results[region_id] = {
            "voxels": np.sum(mask == region_id),
            "volume_mm3": region_volume_mm3,
            "volume_cm3": region_volume_cm3,
        }

    return results


# -------------------
# 3. Shape descriptors
# -------------------
def tumor_shape_metrics(mask_path):
    nii = nib.load(mask_path)
    mask = nii.get_fdata() > 0
    voxel_spacing = nii.header.get_zooms()

    coords = np.array(np.where(mask))
    if coords.size == 0:
        return {"bbox_mm": [0, 0, 0], "longest_axis_mm": 0,
                "shortest_axis_mm": 0, "aspect_ratio": 0}

    min_coords = coords.min(axis=1) * voxel_spacing
    max_coords = coords.max(axis=1) * voxel_spacing
    lengths = max_coords - min_coords

    return {
        "bbox_mm": lengths.tolist(),
        "longest_axis_mm": float(max(lengths)),
        "shortest_axis_mm": float(min(lengths)),
        "aspect_ratio": float(max(lengths) / min(lengths))
    }


# -------------------
# 4. RECIST-like measurement (longest axial diameter)
# -------------------
def recist_longest_diameter(mask_path):
    nii = nib.load(mask_path)
    mask = nii.get_fdata() > 0
    voxel_spacing = nii.header.get_zooms()

    max_diameter = 0
    for z in range(mask.shape[2]):  # iterate over slices
        slice_mask = mask[:, :, z]
        if slice_mask.sum() == 0:
            continue
        props = measure.regionprops(slice_mask.astype(int))
        for prop in props:
            # get longest axis length in this slice (major axis length)
            diameter = prop.major_axis_length * voxel_spacing[0]
            if diameter > max_diameter:
                max_diameter = diameter

    return max_diameter


# -------------------
# 5. Surface area & sphericity
# -------------------
def surface_and_sphericity(mask_path):
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    voxel_spacing = mask_img.header.get_zooms()

    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    if np.sum(mask) == 0:
        # No foreground, return zeros
        return 0.0, 0.0

    verts, faces, _, _ = measure.marching_cubes(mask, level=0, spacing=voxel_spacing)
    surface = measure.mesh_surface_area(verts, faces)
    volume = np.sum(mask) * np.prod(voxel_spacing)
    sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface
    return surface, sphericity


# -------------------
# 6. Batch analysis
# -------------------
def analyze_all(cases):
    data = []
    for case_id, mask_path in cases.items():
        vol_mm3, vol_cm3, spacing = calculate_volume(mask_path)
        shape = tumor_shape_metrics(mask_path)
        recist = recist_longest_diameter(mask_path)
        surface, sphericity = surface_and_sphericity(mask_path)

        row = {
            "case": case_id,
            "volume_mm3": vol_mm3,
            "volume_cm3": vol_cm3,
            "longest_axis_mm": shape["longest_axis_mm"],
            "aspect_ratio": shape["aspect_ratio"],
            "RECIST_diameter_mm": recist,
            "surface_area_mm2": surface,
            "sphericity": sphericity
        }
        data.append(row)

    return pd.DataFrame(data)


def main(fold_num):
    fold_num = int(fold_num)
    label_base = f'fold_{fold_num}/labels'
    pred_base = f'fold_{fold_num}/pred'
    out_base = f'fold_{fold_num}/results/volumetric_analysis'

    os.makedirs(out_base, exist_ok=True)

    # Get available label cases
    label_cases = [f for f in os.listdir(label_base) if f.endswith('.nii.gz')]
    pred_cases = [f for f in os.listdir(pred_base) if f.endswith('.nii.gz')]

    # Build case dictionaries for analysis
    label_dict = {}
    pred_dict = {}

    for label_file in label_cases:
        idx = label_file.split('_')[1].split('.')[0]  # e.g., '0000', '0031'
        label_path = os.path.join(label_base, label_file)
        label_dict[idx] = label_path

    for pred_file in pred_cases:
        idx = pred_file.split('_')[1].split('.')[0]
        pred_path = os.path.join(pred_base, pred_file)
        pred_dict[idx] = pred_path

    # Analyze labels
    print("Analyzing ground truth labels...")
    label_df = analyze_all(label_dict)
    label_df.to_csv(os.path.join(out_base, 'label_volumetric.csv'), index=False)

    # Analyze predictions
    print("Analyzing predictions...")
    pred_df = analyze_all(pred_dict)
    pred_df.to_csv(os.path.join(out_base, 'pred_volumetric.csv'), index=False)

    print("Analysis complete. Results saved to:", out_base)
    
if __name__ == "__main__":
    fold = 1
    main(fold)