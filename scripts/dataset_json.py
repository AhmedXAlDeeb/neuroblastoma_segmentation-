import json
from pathlib import Path

# Set your dataset ID and paths
dataset_id = 1
dataset_name = f"Dataset{dataset_id:03}_Neuroblastoma"
base = Path(r"D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-\data\raw") / dataset_name

# List all training images (with _0000 in the name)
imagesTr = sorted((base / "imagesTr").glob("*.nii.gz"))
labelsTr = sorted((base / "labelsTr").glob("*.nii.gz"))

# Ensure match
assert len(imagesTr) == len(labelsTr), "Mismatch between imagesTr and labelsTr"

json_dict = {
    "channel_names": {
        "0": "CT"  # or "MRI", depending on your case
    },
    "labels": {
        "0": "background",
        "1": "tumor"  # change label class name if needed
    },
    "file_ending": ".nii.gz",
    "numTraining": len(imagesTr),
    "training": [
        {
            "image": f"./imagesTr/{img.name}",
            "label": f"./labelsTr/{lbl.name}"
        }
        for img, lbl in zip(imagesTr, labelsTr)
    ]
}

# Write dataset.json
out_path = base / "dataset.json"
with open(out_path, "w") as f:
    json.dump(json_dict, f, indent=4)

print(f"✅ dataset.json written to: {out_path}")
