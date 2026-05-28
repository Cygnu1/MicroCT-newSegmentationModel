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

import segmentation_models_pytorch as smp
from lovasz_losses import lovasz_hinge

""" Teste da resnext101 implementada (modulada) reduzindo filtros do decoder """

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    data_root: str = r"/home/ari/Alan_araujo/Rochas/data/data_HMP2"  # <- MUDE AQUI
    train_samples: Tuple[str, str] = ("mc3_2_P2",)
    test_sample: str = "mc3_2_P4"


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
    base_decoder_channels: int = 32
    use_aspp: bool = True
    aspp_out_channels: int = 512
    aspp_rates: Tuple[int, int, int] = (6, 12, 18)

    """ Loss """
    loss_name: str = "tversky"   # "focal", "tversky", "dice", "lovasz"
    bce_weight: float = 0.5
    lovasz_weight: float = 0.5
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    focal_gamma: float = 2.0


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
    best_ckpt_name: str = f"Resnext101_ASPP_3_6_9_CBAM__{loss_name}_HMP2_{test_sample}.pt"


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H,W,3)
    return img


def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 127).astype(np.float32)  # robusto p/ 0/255
    return m[..., None]  # (H,W,1)


def resize_hw(x: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    h, w = x.shape[:2]
    if h == size and w == size:
        return x
    x2 = cv2.resize(x, (size, size), interpolation=interp)
    if x2.ndim == 2:
        x2 = x2[..., None]
    return x2


def normalize_image(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
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
class RockPoreSegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: int, in_channels: int, augment: bool = False):
        self.pairs = pairs
        self.img_size = img_size
        self.in_channels = in_channels
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()
        if random.random() < 0.5:
            img = np.flip(img, axis=0).copy()
            msk = np.flip(msk, axis=0).copy()

        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(0, 1)).copy()
            msk = np.rot90(msk, k, axes=(0, 1)).copy()

        return img, msk

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, mp = self.pairs[idx]
        img = read_image(ip, self.in_channels)
        msk = read_mask(mp)

        img = resize_hw(img, self.img_size, is_mask=False)
        msk = resize_hw(msk, self.img_size, is_mask=True)

        img = normalize_image(img)

        if self.augment:
            img, msk = self._augment(img, msk)

        img = np.ascontiguousarray(img)
        msk = np.ascontiguousarray(msk)

        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        msk_t = torch.from_numpy(np.transpose(msk, (2, 0, 1))).float()

        return {"image": img_t, "mask": msk_t}


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