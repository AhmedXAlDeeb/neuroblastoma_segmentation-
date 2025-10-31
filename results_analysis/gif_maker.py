import os
import numpy as np
import nibabel as nib
import imageio

# -------------------------
# Config (set here)
# -------------------------
FOLD_NUM = 1       # change fold number here
CASE_NUM = "021"    # change case number here (e.g. "01", "12")
WINDOW = 340
LEVEL = 50
ALPHA = 0.2
# -------------------------
# Window/Level helpers
# -------------------------
def compute_window_level(scan, mask, default_window=400, default_level=40):
    tumor = scan[mask > 0]
    if tumor.size == 0:
        return default_window, default_level
    level = np.median(tumor)
    iqr = np.percentile(tumor, 75) - np.percentile(tumor, 25)
    window = max(iqr * 2.0, 10.0)
    return float(window), float(level)

def normalize_slice(img_slice, window, level):
    vmin = level - window / 2.0
    vmax = level + window / 2.0
    norm = np.clip((img_slice - vmin) / (vmax - vmin), 0.0, 1.0)
    return norm  # floats in [0,1]

# -------------------------
# Crop functions (Z-axis only)
# -------------------------
def get_z_bounds_from_label(label_volume):
    z_any = np.any(label_volume > 0, axis=(0, 1))
    z_indices = np.where(z_any)[0]
    if z_indices.size == 0:
        return None
    return int(z_indices.min()), int(z_indices.max())

def crop_z_range(volume, zmin, zmax, margin=0):
    zmin = max(zmin - margin, 0)
    zmax = min(zmax + margin, volume.shape[2] - 1)
    return volume[:, :, zmin:zmax+1]

# -------------------------
# Color helpers
# -------------------------
def color_from_name(name):
    n = (name or "").lower()
    if n in ("red", "r", "reds"):
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if n in ("blue", "b", "blues"):
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if n in ("green", "g", "greens"):
        return np.array([0.0, 1.0, 0.0], dtype=float)
    if n in ("yellow", "y", "yellows"):
        return np.array([1.0, 1.0, 0.0], dtype=float)
    return np.array([1.0, 0.0, 0.0], dtype=float)  # fallback red

# -------------------------
# GIF creation
# -------------------------
def make_gifs_for_case(scan_path, mask_path, out_prefix,
                       mask_color='red', mask_alpha=0.35, margin=0, fps=5,
                       use_fixed_wl=True):

    scan_img = nib.load(scan_path)
    mask_img = nib.load(mask_path)

    scan = scan_img.get_fdata()
    mask = (mask_img.get_fdata() > 0).astype(np.uint8)

    z_bounds = get_z_bounds_from_label(mask)
    if z_bounds is None:
        zmin, zmax = 0, scan.shape[2] - 1
    else:
        zmin, zmax = z_bounds

    zmin_crop = max(zmin - margin, 0)
    zmax_crop = min(zmax + margin, scan.shape[2] - 1)

    scan_c = scan[:, :, zmin_crop:zmax_crop + 1]
    mask_c = mask[:, :, zmin_crop:zmax_crop + 1]

    if use_fixed_wl:
        # fixed values from your viewer
        window, level = WINDOW, LEVEL
    else:
        window, level = compute_window_level(scan, mask)

    vol_frames = []
    overlay_frames = []
    color_rgb = color_from_name(mask_color)

    for iz in range(scan_c.shape[2]):
        slice_img = scan_c[:, :, iz]
        slice_mask = (mask_c[:, :, iz] > 0).astype(np.uint8)

        norm = normalize_slice(slice_img, window, level)
        rgb = np.stack([norm, norm, norm], axis=-1)

        overlay_rgb = rgb.copy()
        if slice_mask.sum() > 0:
            overlay_rgb[slice_mask == 1] = (
                (1 - mask_alpha) * rgb[slice_mask == 1] +
                mask_alpha * color_rgb
            )

        vol_frame = (rgb * 255.0).astype(np.uint8)
        overlay_frame = np.clip(overlay_rgb * 255.0, 0, 255).astype(np.uint8)

        vol_frames.append(vol_frame)
        overlay_frames.append(overlay_frame)

    vol_out = out_prefix + "_volume.gif"
    ov_out = out_prefix + "_overlay.gif"
    imageio.mimsave(vol_out, vol_frames, fps=fps)
    imageio.mimsave(ov_out, overlay_frames, fps=fps)
    return vol_out, ov_out

# -------------------------
# Run for single case
# -------------------------
if __name__ == "__main__":
    # build directories from fold/case
    scan_base = f'fold_{FOLD_NUM}/scans'
    pred_base = f'fold_{FOLD_NUM}/postprocessed_pred'
    label_base = f'fold_{FOLD_NUM}/labels'
    out_base = f'fold_{FOLD_NUM}/results/gifs'
    os.makedirs(out_base, exist_ok=True)

    scan_file = f'case_{CASE_NUM}.nii.gz'
    pred_file = f'case_0{CASE_NUM}.nii.gz'
    label_file = f'case_0{CASE_NUM}.nii.gz'
    scan_path = os.path.join(scan_base, scan_file)
    pred_path = os.path.join(pred_base, pred_file)
    label_path = os.path.join(label_base, label_file)
    print(label_path)

    if not os.path.exists(scan_path) or not os.path.exists(label_path):
        print(f"Missing scan or label for case {CASE_NUM}")
    else:
        out_prefix_label = os.path.join(out_base, f'case_{CASE_NUM}_label')
        vol_label, ov_label = make_gifs_for_case(
            scan_path, label_path, out_prefix_label,
            mask_color='green', mask_alpha=ALPHA,
            margin=2, fps=5, use_fixed_wl=True
        )
        print("  saved:", vol_label, ov_label)

    if os.path.exists(pred_path):
        out_prefix_pred = os.path.join(out_base, f'case_{CASE_NUM}_pred')
        vol_pred, ov_pred = make_gifs_for_case(
            scan_path, pred_path, out_prefix_pred,
            mask_color='red', mask_alpha=ALPHA,
            margin=2, fps=5, use_fixed_wl=True
        )
        print("  saved:", vol_pred, ov_pred)
    else:
        print("Prediction missing for", CASE_NUM)
