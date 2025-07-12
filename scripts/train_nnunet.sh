# Neuroblastoma Segmentation

This project applies nnU-Net and custom segmentation models to MRI neuroblastoma datasets.

## Setup

\\\ash
pip install -r requirements.txt
\\\

## Folder Layout

- \data/\: nnU-Net raw, preprocessed, and results
- \models/\: custom architectures (e.g., UNet++)
- \scripts/\: training, prediction, and evaluation
- \utils/\: Dataset and metrics
- \
otebooks/\: EDA and visualizations

## Training with nnU-Net

\\\ash
bash scripts/train_nnunet.sh
\\\

## Training custom models

\\\ash
python scripts/train_custom.py
\\\

## Evaluation

\\\ash
python scripts/evaluate.py
\\\

## Dataset Format

\\\
data/raw/Dataset001_Neuroblastoma/
├── imagesTr/
├── labelsTr/
└── dataset.json
\\\
"@ | Out-File -Encoding utf8 README.md

# ----------------------------
# Create scripts/train_nnunet.sh
# ----------------------------
@"
#!/bin/bash

# Set project-specific variables
export nnUNet_raw_data_base=\D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-/data/raw
export nnUNet_preprocessed=\D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-/data/preprocessed
export RESULTS_FOLDER=\D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-/data/trained_models

DATASET_ID=001
CONFIG=3d_fullres

# Step 1: Plan and preprocess the dataset
echo "Planning and preprocessing..."
nnUNetv2_plan_and_preprocess -d \ -c \

# Step 2: Train all 5 folds
for FOLD in 0 1 2 3 4
do
  echo "Training fold \..."
  nnUNetv2_train -d \ -c \ -f \
done

# Optional: Evaluation step
echo "Evaluating best configuration..."
nnUNetv2_find_best_configuration -d \

# Evaluate one fold (e.g., fold 0)
echo "Running evaluation..."
nnUNetv2_evaluate_folder -d \ -c \ -f 0
