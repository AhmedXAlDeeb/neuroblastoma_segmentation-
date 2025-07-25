# ------------------- FIXED PowerShell Script -------------------

# Set correct nnU-Net v2 environment variables
$env:nnUNet_raw = "D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-\data\raw"
$env:nnUNet_preprocessed = "D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-\data\preprocessed"
$env:nnUNet_results = "D:\Liver Segmentation Meena 2024\Neuroblastoma2026\neuroblastoma_segmentation-\results"

$dataset_id = 001
$config = "3d_fullres"
$folds = 0..4
$pretrained_model_path = ""  # Optional .pth path if fine-tuning

Write-Host "🌐 nnU-Net environment set:"
Write-Host "• nnUNet_raw         = $env:nnUNet_raw"
Write-Host "• nnUNet_preprocessed = $env:nnUNet_preprocessed"
Write-Host "• nnUNet_results     = $env:nnUNet_results"

# ------------------- Preprocessing -------------------
Write-Host "`n🏗️  Running preprocessing..."
nnUNetv2_plan_and_preprocess -d $dataset_id --verify_dataset_integrity

# ------------------- Training -------------------
Write-Host "`n🚀 Starting training..."

foreach ($fold in $folds) {
    Write-Host "`n🧠 Training fold $fold..."
    if ($pretrained_model_path -ne "") {
        nnUNetv2_train $dataset_id $config $fold --pretrained_weights $pretrained_model_path
    } else {
        nnUNetv2_train $dataset_id $config $fold
    }
}

Write-Host "`n✅ All training done. Check: $env:nnUNet_results"
