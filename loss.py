import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lovasz_losses import lovasz_hinge


# -----------------------------
# Loss
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image: bool = False):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits e targets chegam como (B,1,H,W)
        logits = logits.squeeze(1)   # (B,H,W)
        targets = targets.squeeze(1) # (B,H,W)

        # lovasz_hinge espera labels binários
        targets = targets.float()
        return lovasz_hinge(logits, targets, per_image=self.per_image)


class BCELovaszLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, lovasz_weight: float = 0.5, per_image: bool = False):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=per_image)
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.lovasz_weight * self.lovasz(logits, targets)
        )


def build_loss(cfg):
    name = cfg.loss_name.lower()

    if name == "dice":
        return smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
        )

    elif name == "focal":
        return smp.losses.FocalLoss(
            mode="binary",
            gamma=cfg.focal_gamma,
        )

    elif name == "tversky":
        return smp.losses.TverskyLoss(
            mode="binary",
            from_logits=True,
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
        )

    elif name == "lovasz":
        return LovaszHingeLoss(per_image=False)

    elif name == "bce_lovasz":
        return BCELovaszLoss(
            bce_weight=cfg.bce_weight,
            lovasz_weight=cfg.lovasz_weight,
            per_image=False,
        )

    else:
        raise ValueError(f"Loss desconhecida: {cfg.loss_name}")