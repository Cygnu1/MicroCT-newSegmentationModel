import os
import cv2
import numpy as np
from glob import glob
from skimage.exposure import match_histograms
from tqdm import tqdm

# caminho das imagens
pasta_entrada = "D:\\Alan\\Doutorado\\Rochas\\data\\C2D\\Slices_segmented\\original_512"
imagem_referencia_path = "D:\\Alan\\Doutorado\\Rochas\\data\\MC3_2_P2\\Slices_19um\\Slices_19um\\Original_512\\MC2P2_19um_0155.tiff"
pasta_saida = "D:\\Alan\\Doutorado\\Rochas\\data\\C2D\\Slices_segmented\\original_512_HMP2"

os.makedirs(pasta_saida, exist_ok=True)

# lê imagem de referência
ref = cv2.imread(imagem_referencia_path, cv2.IMREAD_UNCHANGED)

# print("shape:", ref.shape)

# print("Tipo: ", ref.dtype)

# print("max: ", np.max(ref))
# print("Min: ", np.min(ref))

# processa imagens da primeira pasta
for img_path in tqdm(glob(os.path.join(pasta_entrada, "*")), desc="Processando imagens..."):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # print("img shape:", img.shape)

    # print("img Tipo: ", img.dtype)

    # print("img max: ", np.max(img))
    # print("img Min: ", np.min(img))

    # histogram matching
    matched = match_histograms(img, ref, channel_axis=None)

    # converter para o tipo original, se necessário
    if np.issubdtype(img.dtype, np.integer):
        matched = np.clip(matched, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)
        matched = matched.astype(img.dtype)

    # print("matched shape:", matched.shape)

    # print("matched Tipo: ", matched.dtype)

    # print("matched max: ", np.max(matched))
    # print("matched Min: ", np.min(matched))

    nome = os.path.basename(img_path)
    cv2.imwrite(os.path.join(pasta_saida, nome), matched)

    # Rodar apenas 1x, teste
    # break
    
print("Processo concluído!! Imagens alteradas")