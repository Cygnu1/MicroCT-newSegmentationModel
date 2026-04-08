import os
import cv2
import numpy as np
from glob import glob
from skimage.exposure import match_histograms
from tqdm import tqdm
import matplotlib.pyplot as plt

# caminho das imagens
pasta_entrada = "D:\\Alan\\Doutorado\\Rochas\\data\\C2D\\Slices_segmented\\original_512"
imagem_referencia_path = "D:\\Alan\\Doutorado\\Rochas\\data\\MC3_2_P2\\Slices_19um\\Slices_19um\\Original_512\\MC2P2_19um_0155.tiff"
pasta_saida = "D:\\Alan\\Doutorado\\Rochas\\data\\C2D\\Slices_segmented\\original_512_HMP2"

os.makedirs(pasta_saida, exist_ok=True)

# lê imagem de referência
ref = cv2.imread(imagem_referencia_path, cv2.IMREAD_UNCHANGED)

# processa imagens da primeira pasta
for img_path in tqdm(glob(os.path.join(pasta_entrada, "*")), desc="Processando imagens..."):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist = (X - cx)**2 + (Y - cy)**2

    raio = 256
    mask = dist <= raio**2

    # Aplicar máscara circular
    img_vals = img[mask]
    ref_vals = ref[mask]

    # Histogram matching só na ROI
    matched_vals = match_histograms(img_vals, ref_vals)

    # Reconstruir imagem
    match = img.copy()
    match[mask] = matched_vals

    # converter para o tipo original, se necessário
    if np.issubdtype(img.dtype, np.integer):
        match = np.clip(match, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)
        match = match.astype(img.dtype)


    plt.imshow(img, cmap='gray')
    plt.show()

    plt.imshow(match, cmap='gray')
    plt.show()

    nome = os.path.basename(img_path)
    # cv2.imwrite(os.path.join(pasta_saida, nome), match)

    # Rodar apenas 1x, teste
    break
    
print("Processo concluído!! Imagens alteradas")