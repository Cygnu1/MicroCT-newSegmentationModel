import torch


class SegmentationMetrics:
    """
    Métricas globais para segmentação binária.

    Acumula TP, FP, FN e TN ao longo de todos
    os batches e calcula as métricas ao final.
    """

    def __init__(
        self,
        threshold=0.5,
        eps=1e-7,
    ):

        self.threshold = threshold
        self.eps = eps

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    # ========================================================
    # UPDATE
    # ========================================================

    @torch.no_grad()
    def update(
        self,
        logits,
        targets,
    ):

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= self.threshold
        )

        targets = (
            targets >= 0.5
        )

        # --------------------------------------------
        # Flatten
        # --------------------------------------------

        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        # --------------------------------------------
        # Confusion matrix
        # --------------------------------------------

        tp = torch.sum(
            predictions & targets
        ).item()

        fp = torch.sum(
            predictions & (~targets)
        ).item()

        fn = torch.sum(
            (~predictions) & targets
        ).item()

        tn = torch.sum(
            (~predictions) & (~targets)
        ).item()

        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)

    # ========================================================
    # COMPUTE
    # ========================================================

    def compute(self):

        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)
        tn = float(self.tn)

        # --------------------------------------------
        # Dice / F1
        # --------------------------------------------

        dice = (
            2.0 * tp
        ) / (
            2.0 * tp
            + fp
            + fn
            + self.eps
        )

        # --------------------------------------------
        # IoU
        # --------------------------------------------

        iou = (
            tp
        ) / (
            tp
            + fp
            + fn
            + self.eps
        )

        # --------------------------------------------
        # Precision
        # --------------------------------------------

        precision = (
            tp
        ) / (
            tp
            + fp
            + self.eps
        )

        # --------------------------------------------
        # Recall
        # --------------------------------------------

        recall = (
            tp
        ) / (
            tp
            + fn
            + self.eps
        )

        # --------------------------------------------
        # Specificity
        # --------------------------------------------

        specificity = (
            tn
        ) / (
            tn
            + fp
            + self.eps
        )

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,

            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }