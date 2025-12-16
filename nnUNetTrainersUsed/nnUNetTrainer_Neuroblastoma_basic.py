# nnUNetTrainer_Neuroblastoma_custom.py
# (The user's file was named this, but the class was '...basic'. I've kept the class name.)

import torch
# Import the new optimizer and scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Neuroblastoma_basic(nnUNetTrainer):
    """
    Simplified custom trainer for Neuroblastoma segmentation.
    Uses default nnUNet loss but improved optimizer and learning rate scheduling.
    
    This version NOW IMPLEMENTS AdamW and CosineAnnealingLR.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # --- Training parameters ---
        
        # 1e-2 is for SGD. For AdamW, a smaller LR is required.
        self.initial_lr = 0.01          # Safe start for AdamW
        self.num_epochs = 50        # Training length
        
        # WARNING: Overriding patch size is NOT recommended.
        # It fights nnU-Net's auto-planning and can worsen overfitting.
        # Let nnU-Net choose the patch size from its plan.
        self.patch_size = (128, 128, 128) 
        
        # self.weight_decay is already set by super().__init__ (default 3e-5)
        # We will use it in configure_optimizers

    def initialize(self):
        """
        Initialize network, dataloaders, and default nnUNet loss (e.g. DiceCE)
        """
        super().initialize()
        # Updated the log message to be accurate
        self.print_to_log_file(
            "Using default nnUNet loss with AdamW optimizer and CosineAnnealingLR scheduler."
        )

    # def configure_optimizers(self):
    #     """
    #     This is the NEW method that actually implements the custom optimizer and scheduler.
    #     """
        
    #     # Setup the optimizer: AdamW
    #     optimizer = AdamW(
    #         self.network.parameters(),
    #         lr=self.initial_lr,
    #         weight_decay=self.weight_decay,
    #         amsgrad=True  # Use AMSGrad variant for more stable training
    #     )

    #     # Setup the scheduler: Cosine Annealing
    #     # This smoothly decreases the LR from initial_lr to eta_min over num_epochs
    #     scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=self.num_epochs,  # Number of epochs to complete one cycle
    #         eta_min=1e-5          # Minimum learning rate
    #     )

    #     return optimizer, scheduler

    def on_epoch_end(self):
        """
        Add custom logging at end of each epoch.
        This method is fine as-is. super().on_epoch_end() will call
        the scheduler.step() for you (if it's not ReduceLROnPlateau).
        """
        super().on_epoch_end()
        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_to_log_file(f"Epoch {self.current_epoch} finished. LR={current_lr:.6f}")