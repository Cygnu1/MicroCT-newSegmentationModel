import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# Caminhos
pasta_gt = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data\c2d\Segmented_512"
pasta_pred = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\results\clahe_c2d\resnext101_32x8d_512_dice_clahe_c2d\pred_bin"
pasta_saida = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\results\clahe_c2d\resnext101_32x8d_512_dice_clahe_c2d\errorMap"

os.makedirs(pasta_saida, exist_ok=True)

# Lista de arquivos
arquivos_gt = sorted(glob(os.path.join(pasta_gt, "*")))

for path_gt in tqdm(arquivos_gt):

    nome = os.path.basename(path_gt)
    path_pred = os.path.join(pasta_pred, nome)

    if not os.path.exists(path_pred):
        print(f"Predição não encontrada para {nome}")
        continue

    # Ler imagens
    gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

    # Binarizar
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    # Criar imagem RGB
    h, w = gt.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    # Modelo segmentou a mais (FP) -> Verde
    false_positive = (pred == 1) & (gt == 0)
    diff[false_positive] = [0,255,0]

    # Modelo segmentou a menos (FN) -> Vermelho
    false_negative = (pred == 0) & (gt == 1)
    diff[false_negative] = [0,0,255]

    # Salvar
    cv2.imwrite(os.path.join(pasta_saida, nome), diff)

    # Rodar apenas uma vez (teste)
    # break
    

print("Finalizado!")