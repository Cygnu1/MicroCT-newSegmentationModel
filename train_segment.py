import os
import glob
from tqdm import tqdm
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *
from model import *
from loss import *

import segmentation_models_pytorch as smp


""" Teste da resnext101 implementada (modulada) reduzindo filtros do decoder """

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_root: str = r"/home/ari/Alan_araujo/Rochas/data/data_HM"  # <- MUDE AQUI
    train_samples: Tuple[str, str] = ("mc3_2_P2","mc3_2_P4")
    test_sample: str = "c2d"

    """ configs """
    img_size: int = 512
    in_channels: int = 3          # 1 se grayscale; 3 se RGB
    num_classes: int = 1          # binário

    """ Architecture """
    encoder_name: str = "resnext101_32x8d"
    encoder_weights: str = "imagenet"  # se in_channels=1, abaixo a gente desliga automaticamente
    attention: str = "scse" # Options  'None' and 'scse'
    use_cbam: bool = True
    cbam_reduction: int = 16
    cbam_spatial_kernel: int = 7
    base_decoder_channels: int = 128
    use_aspp: bool = True
    aspp_out_channels: int = 512
    aspp_rates: Tuple[int, int, int] = (6, 12, 18)

    """ Loss """
    loss_name: str = "tversky"   # "focal", "tversky", "dice", "lovasz"
    bce_weight: float = 0.5
    lovasz_weight: float = 0.5
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_gamma: float = 2.0


    """ Training """
    lr: float = 1e-5
    batch_size: int = 8
    num_workers: int = 2
    max_epochs: int = 200
    augment_train: bool = False
    early_patience: int = 14
    early_min_delta: float = 1e-4
    threshold: float = 0.5  # threshold para binarizar a saída

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    out_dir: str = "./files"
    best_ckpt_name: str = f"Resnext_Limiar{threshold}_CBAM_128filters_Tversky7_{test_sample}.pt"



# -----------------------------
# Metrics via confusion counts
# -----------------------------
@torch.no_grad()
def confusion_counts_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retorna TP, FP, FN, TN somados no batch.
    logits/targets: (B,1,H,W)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    return tp, fp, fn, tn


def safe_div(num: float, den: float, eps: float = 1e-7) -> float:
    return float(num / (den + eps))


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-7) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp, eps)
    recall = safe_div(tp, tp + fn, eps)  # sensibilidade
    iou = safe_div(tp, tp + fp + fn, eps)
    dice = safe_div(2 * tp, 2 * tp + fp + fn, eps)  # == F1
    f1 = dice
    specificity = safe_div(tn, tn + fp, eps)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn, eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "specificity": specificity,
        "accuracy": accuracy,
    }


# -----------------------------
# Early stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)

        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, thr: float = 0.5):
    model.eval()
    total_loss = 0.0

    # acumula contagens globais (micro-average por pixel)
    tp = fp = fn = tn = 0.0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())

        _tp, _fp, _fn, _tn = confusion_counts_from_logits(logits, y, thr=thr)
        tp += float(_tp.cpu().item())
        fp += float(_fp.cpu().item())
        fn += float(_fn.cpu().item())
        tn += float(_tn.cpu().item())

    mets = metrics_from_counts(tp, fp, fn, tn)
    mets["loss"] = total_loss / max(1, len(loader))
    return mets


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = CFG()
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    # --- Monta pares por amostra
    train_pairs = []
    for s in cfg.train_samples:
        train_pairs += list_pairs(os.path.join(cfg.data_root, s))

    test_pairs = list_pairs(os.path.join(cfg.data_root, cfg.test_sample))

    # --- Dataloaders
    train_ds = RockPoreSegDataset(train_pairs, cfg.img_size, cfg.in_channels, augment=cfg.augment_train)
    test_ds = RockPoreSegDataset(test_pairs, cfg.img_size, cfg.in_channels, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # --- Modelo SMP
    encoder_weights = cfg.encoder_weights
    if cfg.in_channels != 3 and cfg.encoder_weights == "imagenet":
        print("[AVISO] in_channels != 3 com encoder_weights='imagenet'. Vou usar encoder_weights=None para evitar conflito.")
        encoder_weights = None


    if cfg.use_cbam == True:
        print("[AVISO] Modelo com CBAM")
        model = ResNeXt101SegmentationModel(
            n_classes=1,
            pretrained=True,
            in_channels=cfg.in_channels,
            base_decoder_channels=cfg.base_decoder_channels,
            use_cbam=cfg.use_cbam,
            cbam_reduction=cfg.cbam_reduction,
            cbam_spatial_kernel=cfg.cbam_spatial_kernel,
            use_aspp=cfg.use_aspp,
            aspp_out_channels=cfg.aspp_out_channels,
            aspp_rates=cfg.aspp_rates,
        ).to(cfg.device)
    else:
        print("[AVISO] Modelo sem CBAM")
        model = ResNeXt101SegmentationModel(
            n_classes=1,
            pretrained=True,
            in_channels=cfg.in_channels,
            base_decoder_channels=cfg.base_decoder_channels,
            use_cbam=cfg.use_cbam,
        ).to(cfg.device)

    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = build_loss(cfg)
    print(f"Usando loss: {cfg.loss_name}")

    # Early stopping baseado em Dice (ou F1, que aqui é igual)
    early = EarlyStopping(patience=cfg.early_patience, min_delta=cfg.early_min_delta, mode="max")

    best_path = os.path.join(cfg.out_dir, cfg.best_ckpt_name)
    best_dice = -1.0

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.device)
        te = evaluate(model, test_loader, loss_fn, cfg.device, thr=cfg.threshold)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} | "
            f"test_loss={te['loss']:.4f} | "
            f"Dice={te['dice']:.4f} | IoU={te['iou']:.4f} | "
            f"Prec={te['precision']:.4f} | Rec={te['recall']:.4f} | F1={te['f1']:.4f}"
        )

        # salva melhor (por Dice)
        if te["dice"] > best_dice:
            best_dice = te["dice"]
            torch.save(
                {"model_state_dict": model.state_dict(), "cfg": cfg.__dict__, "best_dice": best_dice, "epoch": epoch},
                best_path
            )
            print(f"  ✅ Salvou melhor modelo: dice={best_dice:.4f} em {best_path}")

        # early stopping
        if early.step(te["dice"]):
            print(f"  ⏹️ Early stopping ativado. Melhor dice: {early.best:.4f}")
            break

    # Avaliação final carregando o melhor
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_best = evaluate(model, test_loader, loss_fn, cfg.device, thr=cfg.threshold)

    print("\n🏁 Melhor checkpoint (TESTE)")
    print(f"Model Saved: {best_path}")
    print(
        f"Loss={te_best['loss']:.4f} | Dice={te_best['dice']:.4f} | IoU={te_best['iou']:.4f} | "
        f"Prec={te_best['precision']:.4f} | Rec={te_best['recall']:.4f} | F1={te_best['f1']:.4f} | "
        f"Spec={te_best['specificity']:.4f} | Acc={te_best['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()