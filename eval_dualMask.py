import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# Funções de métricas
# ----------------------------
def confusion_counts_from_logits(logits, targets, thr=0.5):
    probs = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    preds = (probs > thr).float()
    targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    return tp, fp, fn, tn

def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    iou = float(tp / (tp + fp + fn + eps))
    dice = float(2 * tp / (2 * tp + fp + fn + eps))
    specificity = float(tn / (tn + fp + eps))
    accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))
    f1 = dice
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "specificity": specificity,
        "accuracy": accuracy
    }

# ----------------------------
# Diretórios das máscaras
# ----------------------------
mask1_dir = Path("C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\results\\small_tversky9\\Resnext_SCSE_tversky9_small_c2d\\pred_bin")
mask2_dir = Path("C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\results\\large_tversky1\\Resnext_SCSE_tversky1_large_c2d\\pred_bin")
gt_dir = Path("C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\c2d\\Segmented_HM_512")
save_dir = Path("C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\results\\combined_small_large_tversky")
save_dir.mkdir(parents=True, exist_ok=True)

# Lista de arquivos (assume mesmo nome nas pastas)
mask_files = sorted(mask1_dir.glob("*.tiff"))

# ----------------------------
# Avaliação e combinação
# ----------------------------
tp = fp = fn = tn = 0

for f in tqdm(mask_files, desc="Processando máscaras"):
    # lê máscaras
    m1 = cv2.imread(str(mask1_dir/f.name), cv2.IMREAD_GRAYSCALE) / 255.0
    m2 = cv2.imread(str(mask2_dir/f.name), cv2.IMREAD_GRAYSCALE) / 255.0
    gt = cv2.imread(str(gt_dir/f.name), cv2.IMREAD_GRAYSCALE) / 255.0

    # binariza
    m1_bin = (m1 > 0.5).astype(np.float32)
    m2_bin = (m2 > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)

    # OR pixel-a-pixel
    combined = np.maximum(m1_bin, m2_bin)

    # salva máscara combinada
    save_path = save_dir / f.name
    cv2.imwrite(str(save_path), (combined*255).astype(np.uint8))

    # calcula métricas
    _tp, _fp, _fn, _tn = confusion_counts_from_logits(torch.tensor(combined[None, None]), 
                                                       torch.tensor(gt_bin[None, None]), thr=0.5)
    tp += _tp
    fp += _fp
    fn += _fn
    tn += _tn

# métricas finais
metrics = metrics_from_counts(tp, fp, fn, tn)
print("Métricas da combinação OR:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")