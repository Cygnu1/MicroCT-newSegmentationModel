import os
import glob
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_root: str = r"/home/ari/Alan_araujo/Rochas/data/data_HM"     # <- MUDE AQUI
    test_sample: str = "mc3_2_P4"            # <- MUDE AQUI

    ckpt_path: str = r"./files/resnext101_32x8d_512_dice_HMP4_P4.pt"  # <- MUDE AQUI (checkpoint do treino)

    img_size: int = 512
    in_channels: int = 3          # 1 grayscale; 3 RGB
    num_classes: int = 1

    encoder_name: str = "resnext101_32x8d"
    encoder_weights: str = None   # no teste, normalmente None; pesos vêm do ckpt
    threshold: float = 0.5
    attention: str = "scse" # Options  'None' and 'scse'

    batch_size: int = 4
    num_workers: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir: str = "results/resnext101_32x8d_512_dice_HMP4_P4"


# -----------------------------
# I/O helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_pairs(sample_dir: str) -> List[Tuple[str, str]]:
    img_dir = os.path.join(sample_dir, "Original_512")
    msk_dir = os.path.join(sample_dir, "Segmented_512")

    imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
    pairs = []
    for ip in imgs:
        name = os.path.basename(ip)
        mp = os.path.join(msk_dir, name)
        if os.path.exists(mp):
            pairs.append((ip, mp))
        else:
            # Se quiser permitir teste sem máscara, comente esse else e use pairs.append((ip, None))
            pass

    if not pairs:
        raise FileNotFoundError(f"Nenhum par imagem/máscara encontrado em: {sample_dir}")
    return pairs


def read_image(path: str, in_channels: int) -> np.ndarray:
    if in_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = img[..., None]  # (H,W,1)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 127).astype(np.float32)
    return m[..., None]  # (H,W,1)


def resize_hw(x: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    x2 = cv2.resize(x, (size, size), interpolation=interp)
    if x2.ndim == 2:
        x2 = x2[..., None]
    return x2


def normalize_image(x: np.ndarray) -> np.ndarray:
    # normalização simples [0,1] (mesma do treino)
    x = x.astype(np.float32)
    if x.max() > 1.0:
        # cuidado: se for uint16 real, o ideal seria /65535
        x = x / 255.0
    return x


def to_uint8_img(img01: np.ndarray) -> np.ndarray:
    # img01 esperado em [0,1]
    x = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    return x


# -----------------------------
# Dataset
# -----------------------------
class RockPoreSegTestDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: int, in_channels: int):
        self.pairs = pairs
        self.img_size = img_size
        self.in_channels = in_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, mp = self.pairs[idx]
        img = read_image(ip, self.in_channels)
        msk = read_mask(mp)

        # guarda também a imagem original (para overlay)
        img_orig = img.copy()

        img = resize_hw(img, self.img_size, is_mask=False)
        msk = resize_hw(msk, self.img_size, is_mask=True)

        img = normalize_image(img)

        img = np.ascontiguousarray(img)
        msk = np.ascontiguousarray(msk)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        msk_t = torch.from_numpy(msk.transpose(2, 0, 1)).float()

        return {
            "image": img_t,
            "mask": msk_t,
            "img_path": ip,
            "mask_path": mp,
            "img_orig": img_orig,  # (H,W,C) antes do resize
        }


# -----------------------------
# Loss + Metrics (contagens globais)
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


@torch.no_grad()
def confusion_counts_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float):
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


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-7):
    precision = safe_div(tp, tp + fp, eps)
    recall = safe_div(tp, tp + fn, eps)
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
# Visual saving
# -----------------------------
def make_overlay(img: np.ndarray, pred_bin: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    img: (H,W,1) ou (H,W,3) uint8
    pred_bin: (H,W) uint8 0/255
    Retorna overlay BGR uint8 para salvar com cv2.imwrite.
    """
    if img.ndim == 3 and img.shape[2] == 1:
        base = img[..., 0]
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        # img é RGB -> converte para BGR para salvar
        base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # cria máscara colorida (vermelho)
    mask_color = np.zeros_like(base)
    mask_color[:, :, 2] = (pred_bin > 0).astype(np.uint8) * 255

    overlay = cv2.addWeighted(base, 1.0, mask_color, alpha, 0.0)
    return overlay


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = CFG()
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "pred_bin"))
    ensure_dir(os.path.join(cfg.out_dir, "pred_prob"))
    ensure_dir(os.path.join(cfg.out_dir, "overlay"))

    # dataset
    test_dir = os.path.join(cfg.data_root, cfg.test_sample)
    pairs = list_pairs(test_dir)
    ds = RockPoreSegTestDataset(pairs, cfg.img_size, cfg.in_channels)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # model (mesma arquitetura do treino)
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,  # normalmente None
        decoder_attention_type=cfg.attention,
        in_channels=cfg.in_channels,
        classes=cfg.num_classes,
        activation=None,
    ).to(cfg.device)

    # load ckpt
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    total_loss = 0.0
    tp = fp = fn = tn = 0.0

    # lista de (nome_arquivo, f1)
    per_image_f1 = []  

    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["image"].to(cfg.device)
            y = batch["mask"].to(cfg.device)

            logits = model(x)
            loss = 0.5 * bce_loss(logits, y) + 0.5 * dice_loss_fn(logits, y)
            total_loss += float(loss.item())

            _tp, _fp, _fn, _tn = confusion_counts_from_logits(logits, y, cfg.threshold)
            tp += float(_tp.cpu().item())
            fp += float(_fp.cpu().item())
            fn += float(_fn.cpu().item())
            tn += float(_tn.cpu().item())

            # salvar predições por item no batch
            probs = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
            probs = probs[:, 0, :, :]  # (B,H,W)

            for i in range(probs.shape[0]):
                img_path = batch["img_path"][i]
                name = os.path.basename(img_path)

                prob = probs[i]  # (H,W) float [0,1]
                pred_prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
                pred_bin_u8 = ((prob > cfg.threshold).astype(np.uint8) * 255)

                # --------- calcular métricas por imagem ----------
                gt = batch["mask"][i].cpu().numpy()[0]  # (H,W)
                pred = (prob > cfg.threshold).astype(np.float32)

                tp_i = np.sum((pred == 1) & (gt == 1))
                fp_i = np.sum((pred == 1) & (gt == 0))
                fn_i = np.sum((pred == 0) & (gt == 1))

                f1_i = (2 * tp_i) / (2 * tp_i + fp_i + fn_i + 1e-7)

                per_image_f1.append((name, f1_i))

                # salva prob e binário no tamanho de avaliação (img_size x img_size)
                # cv2.imwrite(os.path.join(cfg.out_dir, "pred_prob", name), pred_prob_u8)
                cv2.imwrite(os.path.join(cfg.out_dir, "pred_bin", name), pred_bin_u8)

                # overlay: usa a imagem original, mas vamos redimensionar o pred para o tamanho original
                img_orig = batch["img_orig"][i].numpy() if hasattr(batch["img_orig"][i], "numpy") else batch["img_orig"][i]
                # img_orig é numpy (H,W,C) vindo do dataset (antes do resize)
                H0, W0 = img_orig.shape[:2]
                pred_bin_orig = cv2.resize(pred_bin_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)
                overlay = make_overlay(img_orig, pred_bin_orig, alpha=0.4)
                cv2.imwrite(os.path.join(cfg.out_dir, "overlay", name), overlay)

    avg_loss = total_loss / max(1, len(loader))
    mets = metrics_from_counts(tp, fp, fn, tn)
    mets["loss"] = avg_loss

    print("\nResultados na amostra de teste:")
    print(
        f"Loss={mets['loss']:.4f} | Dice={mets['dice']:.4f} | IoU={mets['iou']:.4f} | "
        f"Precision={mets['precision']:.4f} | Recall={mets['recall']:.4f} | F1={mets['f1']:.4f} | "
        f"Spec={mets['specificity']:.4f} | Acc={mets['accuracy']:.4f}"
    )

    # --------- melhor e pior imagem ----------
    if per_image_f1:
        best_img = max(per_image_f1, key=lambda x: x[1])
        worst_img = min(per_image_f1, key=lambda x: x[1])

        print("\nAvaliação por imagem:")
        print(f"Melhor F1  : {best_img[0]}  | F1 = {best_img[1]:.4f}")
        print(f"Pior F1   : {worst_img[0]} | F1 = {worst_img[1]:.4f}")

    print(f"Modelo avaliado: {cfg.ckpt_path}")
    print(f"\nPredições salvas em: {os.path.abspath(cfg.out_dir)}")
    print("  - pred_prob/  (probabilidade 0–255)")
    print("  - pred_bin/   (máscara binária 0/255)")
    print("  - overlay/    (visualização por cima da imagem)")


if __name__ == "__main__":
    main()
