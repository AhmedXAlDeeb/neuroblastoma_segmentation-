import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw

def main():
    # -------------------------------
    # 1. Define Dataset and Model
    # -------------------------------
    dataset_id = 'Dataset001_Neuroblastoma'  # change if needed
    model_folder_name = 'STUNetTrainer_large_ft__nnUNetPlans__3d_fullres'
    checkpoint_name = 'checkpoint_final.pth'

    # -------------------------------
    # 2. Define Paths
    # -------------------------------
    model_folder = r'C:\Neuroblastoma\Neuroblastoma\neuroblastoma_segmentation\data\nnUNet_results\Dataset001_Neuroblastoma\STUNetTrainer_large_ft__nnUNetPlans__3d_fullres'
    input_folder = r"C:\Neuroblastoma\Neuroblastoma\neuroblastoma_segmentation\data\nnUNet_raw\Dataset001_Neuroblastoma\imagesTs"
    output_folder =  r"C:\Neuroblastoma\Neuroblastoma\neuroblastoma_segmentation\data\test\prediction"  
    # -------------------------------
    # 3. Initialize Predictor
    # -------------------------------
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    # -------------------------------
    # 4. Load Model
    # -------------------------------
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),  # you can change to (0, 1, 2, 3, 4) if you trained all folds
        checkpoint_name=checkpoint_name
    )

    # -------------------------------
    # 5. Run Prediction
    # -------------------------------
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    print(f"Inference complete! Predictions saved to: {output_folder}")

if __name__ == "__main__":
    main()
