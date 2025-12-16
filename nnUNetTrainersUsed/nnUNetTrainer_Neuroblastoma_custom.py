# nnUNetTrainer_Neuroblastoma_custom.py
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi

# Import base nnUNet trainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# -------------------------
# Loss implementations
# -------------------------
class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss:
    Tversky index: TI = TP / (TP + alpha * FN + beta * FP)
    Focal Tversky: (1 - TI) ** gamma
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=4/3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, probs: torch.Tensor, target_onehot: torch.Tensor):
        """
        probs: (B, C, D, H, W) softmax probabilities
        target_onehot: (B, C, D, H, W) one-hot targets
        """
        assert probs.shape == target_onehot.shape, \
            f"Shape mismatch: probs {probs.shape}, target {target_onehot.shape}"

        # Clamp to avoid log(0) or div by 0
        probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)

        dims = (2, 3, 4)
        tp = (probs * target_onehot).sum(dims)
        fp = (probs * (1 - target_onehot)).sum(dims)
        fn = ((1 - probs) * target_onehot).sum(dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        tversky = torch.clamp(tversky, 0, 1)

        loss = (1 - tversky) ** self.gamma
        return loss.mean()


def gaussian_soft_label_from_hard(label_arr: np.ndarray, sigma_voxels: float = 1.0) -> np.ndarray:
    """
    Converts hard label array into soft label array using Gaussian smoothing.
    """
    labels = np.asarray(label_arr, dtype=np.int32)
    C = int(labels.max()) + 1
    soft = np.zeros((C,) + labels.shape, dtype=np.float32)

    for c in range(C):
        mask = (labels == c).astype(np.float32)
        sm = ndi.gaussian_filter(mask, sigma=sigma_voxels) if sigma_voxels > 0 else mask
        soft[c] = sm

    ssum = soft.sum(axis=0, keepdims=True)
    ssum[ssum == 0] = 1.0
    return soft / ssum


class MumfordShahSurrogateLoss(nn.Module):
    """
    Surrogate for Mumford-Shah energy:
      - data fidelity: ||u - y||^2
      - smoothness: lambda_edge * |grad u|^2
    """
    def __init__(self, lambda_edge=0.1, lambda_data=1.0):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_data = lambda_data

    def forward(self, probs: torch.Tensor, target_soft: torch.Tensor,
                image: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Clamp for stability
        probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)

        # Fidelity
        fidelity = F.mse_loss(probs, target_soft, reduction='mean') * self.lambda_data

        # Smoothness (finite diffs)
        dx = probs[..., 1:, :, :] - probs[..., :-1, :, :]
        dy = probs[..., :, 1:, :] - probs[..., :, :-1, :]
        dz = probs[..., :, :, 1:] - probs[..., :, :, :-1]
        grad2 = dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()

        return fidelity + self.lambda_edge * grad2


class CombinedLoss(nn.Module):
    """
    Combination of Focal Tversky Loss + Mumford-Shah surrogate.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0,
                 ft_weight=1.0, ms_weight=0.5,
                 ms_lambda_edge=0.05, ms_lambda_data=1.0):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.ms = MumfordShahSurrogateLoss(lambda_edge=ms_lambda_edge, lambda_data=ms_lambda_data)
        self.ft_weight = ft_weight
        self.ms_weight = ms_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # unwrap if target is list
        if isinstance(target, (list, tuple)):
            target = target[0]

        # One-hot conversion
        if target.ndim == 5 and target.shape[1] == 1:
            t = target.squeeze(1).long()
            C = net_output[0].shape[1] if isinstance(net_output, (tuple, list)) else net_output.shape[1]
            target_onehot = F.one_hot(t, num_classes=C).permute(0, 4, 1, 2, 3).float()
        elif target.ndim == 5 and target.shape[1] > 1:
            target_onehot = target.float()
        else:
            raise ValueError(f"Unexpected target shape: {target.shape}")

        # Deep supervision
        if isinstance(net_output, (tuple, list)):
            n = len(net_output)
            weights = torch.tensor([1 / (2 ** i) for i in range(n)],
                                   device=net_output[0].device,
                                   dtype=net_output[0].dtype)
            weights = weights / weights.sum()

            total = 0.0
            for w, out in zip(weights.cpu().numpy().tolist(), net_output):
                probs = F.softmax(out, dim=1)
                target_up = F.interpolate(target_onehot, size=probs.shape[2:], mode='nearest') \
                    if probs.shape != target_onehot.shape else target_onehot

                ft_loss = self.ft(probs, target_up)
                ms_loss = self.ms(probs, target_up)
                total_loss = self.ft_weight * ft_loss + self.ms_weight * ms_loss

                # Print per-output logs
                print(f"[DeepSup] FT={ft_loss.item():.4f}, MS={ms_loss.item():.4f}, Total={total_loss.item():.4f}")

                total += w * total_loss
            return total
        else:
            probs = F.softmax(net_output, dim=1)
            target_up = F.interpolate(target_onehot, size=probs.shape[2:], mode='nearest') \
                if probs.shape != target_onehot.shape else target_onehot

            ft_loss = self.ft(probs, target_up)
            ms_loss = self.ms(probs, target_up)
            total_loss = self.ft_weight * ft_loss + self.ms_weight * ms_loss

            # Print logs
            print(f"FT={ft_loss.item():.4f}, MS={ms_loss.item():.4f}, Total={total_loss.item():.4f}")

            return total_loss


# -------------------------
# Trainer subclass
# -------------------------
class nnUNetTrainer_Neuroblastoma_custom(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # Loss hyperparameters
        self.loss_alpha = 0.5
        self.loss_beta = 0.5
        self.loss_gamma = 1.0
        self.loss_ft_weight = 1.0
        self.loss_ms_weight = 0.5
        self.loss_ms_lambda_edge = 0.05
        self.loss_ms_lambda_data = 1.0

        # Training hyperparameters
        self.patch_size = (96, 128, 128)
        self.num_epochs = 100
        self.initial_lr = 1e-4   # 🔑 Lowered to avoid NaN

    def initialize(self) -> None:
        super().initialize()
        self.loss = CombinedLoss(
            alpha=self.loss_alpha, beta=self.loss_beta, gamma=self.loss_gamma,
            ft_weight=self.loss_ft_weight, ms_weight=self.loss_ms_weight,
            ms_lambda_edge=self.loss_ms_lambda_edge, ms_lambda_data=self.loss_ms_lambda_data
        )
        self.print_to_log_file("Using CombinedLoss: FocalTversky + Mumford-Shah surrogate")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=getattr(self, 'weight_decay', 0.0)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return optimizer, scheduler

    def precompute_smoothed_labels(self, sigma_voxels=1.0, out_folder=None):
        raise NotImplementedError("Run label smoothing as preprocessing, not in trainer.")
