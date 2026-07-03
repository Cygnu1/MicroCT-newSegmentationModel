import cv2
import numpy as np
import os
import glob
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

import torch

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
