import os
import glob
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import segmentation_models_pytorch as smp


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_root: str = r"/home/ari/Alan_araujo/Rochas/data/data_HMP2_roi"     # <- MUDE AQUI
    test_sample: str = "c2d"            # <- MUDE AQUI

    ckpt_path: str = r"./files/Resnext101_ASPP_3_6_9_CBAM_64Filters_tversky_HMP2_c2d.pt"  # <- MUDE AQUI (checkpoint do treino)

    img_size: int = 512
    in_channels: int = 3          # 1 grayscale; 3 RGB
    num_classes: int = 1

    encoder_name: str = "resnext101_32x8d"
    encoder_weights: str = None   # no teste, normalmente None; pesos vêm do ckpt
    threshold: float = 0.5
    attention: str = "scse" # Options  'None' and 'scse'
    use_cbam: bool = True
    cbam_reduction: int = 16
    cbam_spatial_kernel: int = 7
    base_decoder_channels: int = 64
    use_aspp: bool = True
    aspp_out_channels: int = 512
    aspp_rates: Tuple[int, int, int] = (3, 6, 9)

    batch_size: int = 4
    num_workers: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir: str = "results/Resnext101_ASPP_3_6_9_CBAM_64Filters_tversky_HMP2_c2d"


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
# Utils
# -----------------------------
def make_decoder_channels(base: int = 64, n_blocks: int = 4):
    """
    Gera canais do decoder do mais profundo para o mais raso.

    Ex:
        base=64 -> [512, 256, 128, 64]
        base=32 -> [256, 128, 64, 32]
    """
    return [base * (2 ** i) for i in reversed(range(n_blocks))]

# -----------------------------
# CBAM
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
    
# -----------------------------
# ASPP
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512, rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)

        x = torch.cat([b1, b2, b3, b4, gp], dim=1)
        return self.project(x)

# -----------------------------
# Encoder
# -----------------------------
class ResNeXt101Encoder(nn.Module):
    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()

        weights = None
        if pretrained:
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT

        backbone = models.resnext101_32x8d(weights=weights)

        # adapta conv1 se a entrada não for RGB
        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            if pretrained:
                if in_channels == 1:
                    # média dos pesos RGB -> 1 canal
                    backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1   # 256
        self.layer2 = backbone.layer2   # 512
        self.layer3 = backbone.layer3   # 1024
        self.layer4 = backbone.layer4   # 2048

    def forward(self, x):
        f0 = self.relu(self.bn1(self.conv1(x)))   # ~ 1/2 resolução, 64 canais
        f1 = self.maxpool(f0)                     # ~ 1/4 resolução, 64 canais
        f2 = self.layer1(f1)                      # 256 canais
        f3 = self.layer2(f2)                      # 512 canais
        f4 = self.layer3(f3)                      # 1024 canais
        f5 = self.layer4(f4)                      # 2048 canais

        return [f0, f1, f2, f3, f4, f5]


# -----------------------------
# Decoder block
# -----------------------------
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv1 = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(
                out_channels,
                reduction=cbam_reduction,
                spatial_kernel=cbam_spatial_kernel
            )

    def forward(self, x, skip=None):
        x = self.upconv(x)

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.use_cbam:
            x = self.cbam(x)

        return x


# -----------------------------
# Full model
# -----------------------------
class ResNeXt101SegmentationModel(nn.Module):
    def __init__(
        self,
        n_classes: int = 1,
        pretrained: bool = True,
        in_channels: int = 1,
        base_decoder_channels: int = 64,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
        use_aspp: bool = True,
        aspp_out_channels: int = 512,
        aspp_rates: Tuple[int, int, int] = (6, 12, 18),
    ):
        super().__init__()

        self.encoder = ResNeXt101Encoder(
            pretrained=pretrained,
            in_channels=in_channels
        )

        self.use_aspp = use_aspp

        if self.use_aspp:
            self.aspp = ASPP(
                in_channels=2048,
                out_channels=aspp_out_channels,
                rates=aspp_rates
            )
            bottleneck_channels = aspp_out_channels
        else:
            bottleneck_channels = 2048

        # base=64 -> [512, 256, 128, 64]
        d4, d3, d2, d1 = make_decoder_channels(base=base_decoder_channels, n_blocks=4)

        self.decoder4 = DecoderBlock(
            in_channels=bottleneck_channels,
            skip_channels=1024,
            out_channels=d4,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.decoder3 = DecoderBlock(
            in_channels=d4,
            skip_channels=512,
            out_channels=d3,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.decoder2 = DecoderBlock(
            in_channels=d3,
            skip_channels=256,
            out_channels=d2,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )
        self.decoder1 = DecoderBlock(
            in_channels=d2,
            skip_channels=64,
            out_channels=d1,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel,
        )

        self.final_up = nn.ConvTranspose2d(d1, d1, kernel_size=2, stride=2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(d1 + 64, d1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d1, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        f0, f1, f2, f3, f4, f5 = self.encoder(x)

        if self.use_aspp:
            f5 = self.aspp(f5)

        x = self.decoder4(f5, skip=f4)
        x = self.decoder3(x,  skip=f3)
        x = self.decoder2(x,  skip=f2)
        x = self.decoder1(x,  skip=f0)

        x = self.final_up(x)

        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(
                x,
                size=f0.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, f0], dim=1)
        x = self.final_conv(x)

        # garante mesma resolução da entrada/máscara
        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

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


    if cfg.use_cbam == True:
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
        model = ResNeXt101SegmentationModel(
            n_classes=1,
            pretrained=True,
            in_channels=cfg.in_channels,
            base_decoder_channels=cfg.base_decoder_channels,
            use_cbam=cfg.use_cbam,
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
