import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

IMAGE_FOLDER = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HM\mc3_2_P2\Original_HM_512"
MASK_FOLDER  = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HM\mc3_2_P2\Segmented_HM_512"

OUTPUT_IMAGE_FOLDER = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HM\mc3_2_P2\Original_HM_512_crop"
OUTPUT_MASK_FOLDER  = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HM\mc3_2_P2\Segmented_HM_512_crop"

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)

# Escolha o raio que deseja manter
# Exemplo: se o cilindro possui raio 250 px e deseja remover
# 5 pixels da borda, utilize 245.
RADIUS = 254

# Valor do fundo (normalmente 0)
BACKGROUND_VALUE = 0

# ==========================================================
# CRIA MÁSCARA CIRCULAR
# ==========================================================

# Lê uma imagem apenas para descobrir o tamanho
sample = cv2.imread(
    os.path.join(IMAGE_FOLDER, os.listdir(IMAGE_FOLDER)[0]),
    cv2.IMREAD_UNCHANGED,
)

h, w = sample.shape[:2]

center = (w // 2, h // 2)

mask = np.zeros((h, w), dtype=np.uint8)

cv2.circle(
    mask,
    center,
    RADIUS,
    255,
    thickness=-1,
)

mask_bool = mask.astype(bool)

# ==========================================================
# PROCESSAMENTO
# ==========================================================

files = sorted(os.listdir(IMAGE_FOLDER))

for file in tqdm(files):

    image = cv2.imread(
        os.path.join(IMAGE_FOLDER, file),
        cv2.IMREAD_UNCHANGED,
    )

    label = cv2.imread(
        os.path.join(MASK_FOLDER, file),
        cv2.IMREAD_UNCHANGED,
    )

    image_out = image.copy()
    label_out = label.copy()

    # imagem
    if image.ndim == 2:
        image_out[~mask_bool] = BACKGROUND_VALUE
    else:
        image_out[~mask_bool, :] = BACKGROUND_VALUE

    # máscara
    if label.ndim == 2:
        label_out[~mask_bool] = 0
    else:
        label_out[~mask_bool, :] = 0

    cv2.imwrite(
        os.path.join(OUTPUT_IMAGE_FOLDER, file),
        image_out,
    )

    cv2.imwrite(
        os.path.join(OUTPUT_MASK_FOLDER, file),
        label_out,
    )


print("Concluído!")