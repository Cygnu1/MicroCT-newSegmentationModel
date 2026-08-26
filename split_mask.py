import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

INPUT_FOLDER = r"C:/Users/alanaraujo/Documents/tecgraf/Doutorado/Rocha/Segmentation/data_HM/mc3_2_P2/Segmented_HM_512_crop"

OUTPUT_SMALL = r"C:/Users/alanaraujo/Documents/tecgraf/Doutorado/Rocha/Segmentation/data_HM/mc3_2_P2/Segmented_HM_512_crop_small"
OUTPUT_LARGE = r"C:/Users/alanaraujo/Documents/tecgraf/Doutorado/Rocha/Segmentation/data_HM/mc3_2_P2/Segmented_HM_512_crop_large"

os.makedirs(OUTPUT_SMALL, exist_ok=True)
os.makedirs(OUTPUT_LARGE, exist_ok=True)

# área limite
AREA_THRESHOLD = 10

# conectividade
CONNECTIVITY = 8

# ==========================================================
# PROCESSAMENTO
# ==========================================================

files = sorted(os.listdir(INPUT_FOLDER))

for file in tqdm(files):

    path = os.path.join(INPUT_FOLDER, file)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    # garante binária
    binary = (mask > 0).astype(np.uint8)

    # componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=CONNECTIVITY
    )

    mask_small = np.zeros_like(mask)
    mask_large = np.zeros_like(mask)

    # label 0 = fundo
    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area <= AREA_THRESHOLD:
            mask_small[labels == label] = 255
        else:
            mask_large[labels == label] = 255

    cv2.imwrite(
        os.path.join(OUTPUT_SMALL, file),
        mask_small
    )

    cv2.imwrite(
        os.path.join(OUTPUT_LARGE, file),
        mask_large
    )

print("Finalizado!")