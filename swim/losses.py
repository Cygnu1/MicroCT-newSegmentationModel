import torch
import torch.nn as nn


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss para segmentação binária.

    Espera:
        logits  -> saída bruta do modelo
        targets -> máscara binária {0, 1}

    O sigmoid é aplicado internamente.
    """

    def __init__(
        self,
        smooth=1.0,
        eps=1e-7,
    ):
        super().__init__()

        self.smooth = smooth
        self.eps = eps

    def forward(
        self,
        logits,
        targets,
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.float()
        targets = targets.float()

        # --------------------------------------------
        # Calcula Dice considerando todo o batch
        # --------------------------------------------

        intersection = torch.sum(
            probabilities * targets
        )

        probability_sum = torch.sum(
            probabilities
        )

        target_sum = torch.sum(
            targets
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probability_sum
            + target_sum
            + self.smooth
            + self.eps
        )

        return 1.0 - dice


# ============================================================
# BCE + DICE
# ============================================================

class BCEDiceLoss(nn.Module):
    """
    Combinação:

        total_loss =
            BCE_WEIGHT * BCE
            +
            DICE_WEIGHT * Dice

    Baseline:
        BCE_WEIGHT  = 1.0
        DICE_WEIGHT = 1.0
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets,
    ):

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


# ============================================================
# TVERSKY
# ============================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss para segmentação binária.

    Formula:

        Tversky =
            (TP + smooth)
            /
            (TP + alpha * FN + beta * FP + smooth)

    Neste projeto:

        alpha = 3.0
        beta  = 1.0

    Portanto, falsos negativos são penalizados
    3x mais que falsos positivos.
    """

    def __init__(
        self,
        alpha=3.0,
        beta=1.0,
        smooth=1.0,
        eps=1e-7,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(
        self,
        logits,
        targets,
    ):

        # Logits -> probabilidades
        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.float()
        targets = targets.float()

        # ----------------------------------------------------
        # True Positive
        # ----------------------------------------------------

        tp = torch.sum(
            probabilities * targets
        )

        # ----------------------------------------------------
        # False Negative
        #
        # GT = 1
        # prediction -> 0
        # ----------------------------------------------------

        fn = torch.sum(
            (1.0 - probabilities)
            * targets
        )

        # ----------------------------------------------------
        # False Positive
        #
        # GT = 0
        # prediction -> 1
        # ----------------------------------------------------

        fp = torch.sum(
            probabilities
            * (1.0 - targets)
        )

        # ----------------------------------------------------
        # Tversky Index
        # ----------------------------------------------------

        tversky = (
            tp + self.smooth
        ) / (
            tp
            + self.alpha * fn
            + self.beta * fp
            + self.smooth
            + self.eps
        )

        return 1.0 - tversky