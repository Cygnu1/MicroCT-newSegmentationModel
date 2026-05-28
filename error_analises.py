import numpy as np
import cv2
from scipy.ndimage import label
import matplotlib.pyplot as plt



# ==========================================================
# 1. Carregar imagens
# ==========================================================
pasta_gt = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HMP2_roi\c2d\Segmentation_512"
pasta_pred = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\results\3samples\Resnext101_128Filters_True_3samples_tversky_HMP2_c2d\pred_bin"

# Altere os caminhos abaixo
pred_path = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\results\3samples\Resnext101_128Filters_True_3samples_tversky_HMP2_c2d\pred_bin\C2D_2738.tiff"
gt_path = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HMP2_roi\c2d\Segmentation_512\C2D_2738.tiff"

pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

if pred is None:
    raise FileNotFoundError(f"Não foi possível carregar: {pred_path}")

if gt is None:
    raise FileNotFoundError(f"Não foi possível carregar: {gt_path}")

if pred.shape != gt.shape:
    raise ValueError("Predição e Ground Truth precisam ter o mesmo tamanho.")


# ==========================================================
# 2. Binarizar imagens
# ==========================================================
# Considera pixels > 127 como poro

pred_bin = pred > 127
gt_bin = gt > 127



# ==========================================================
# 3. Calcular erros
# ==========================================================

# Adicionados pelo modelo: predição marcou poro, GT não
added = (pred_bin == 1) & (gt_bin == 0)

# Retirados pelo modelo: GT era poro, predição não marcou
removed = (pred_bin == 0) & (gt_bin == 1)

plt.imshow(added, cmap='gray')
plt.show()

plt.imshow(removed, cmap='gray')
plt.show()

# ==========================================================
# 4. Encontrar componentes conectados
# ==========================================================

# Conectividade 8-vizinhos
structure = np.ones((3, 3), dtype=np.int8)

labeled_added, num_added = label(added, structure=structure)
labeled_removed, num_removed = label(removed, structure=structure)


# ==========================================================
# 5. Medir tamanho dos grupos
# ==========================================================

sizes_added = np.bincount(labeled_added.ravel())[1:]
sizes_removed = np.bincount(labeled_removed.ravel())[1:]


# ==========================================================
# 6. Classificar tamanhos
# ==========================================================

def classify_sizes(sizes):
    result = {
        "1 pixel": 0,
        "2 a 4 pixels": 0,
        "5 a 9 pixels": 0,
        "10 a 25 pixels": 0,
        "maior que 25 pixels": 0
    }

    for s in sizes:
        if s == 1:
            result["1 pixel"] += 1
        elif s <= 4:
            result["2 a 4 pixels"] += 1
        elif s <= 9:
            result["5 a 9 pixels"] += 1
        elif s <= 25:
            result["10 a 25 pixels"] += 1
        else:
            result["maior que 25 pixels"] += 1

    return result


added_classification = classify_sizes(sizes_added)
removed_classification = classify_sizes(sizes_removed)


# ==========================================================
# 7. Exibir resultados
# ==========================================================

print("\n==============================")
print("ERROS ADICIONADOS PELO MODELO")
print("==============================")
print(f"Total de grupos adicionados: {num_added}")
print(f"Total de pixels adicionados: {np.sum(added)}")
print("Classificação por tamanho:")

for k, v in added_classification.items():
    print(f"{k}: {v}")


print("\n==============================")
print("ERROS RETIRADOS PELO MODELO")
print("==============================")
print(f"Total de grupos retirados: {num_removed}")
print(f"Total de pixels retirados: {np.sum(removed)}")
print("Classificação por tamanho:")

for k, v in removed_classification.items():
    print(f"{k}: {v}")