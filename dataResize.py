import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


if "__main__" == __name__:

    # Tamanho para redimensionar para quadrado
    size = 512

    # Pasta de entrada e saída
    pasta_entrada = r"D:\Alan\Doutorado\Rochas\data\MC3_3_P4\Segmented_HM"   # substitua pelo seu caminho real
    pasta_saida = r"D:\Alan\Doutorado\Rochas\data\MC3_3_P4\Segmented_HM_512" 

    # Cria a pasta de saída se não existir
    os.makedirs(pasta_saida, exist_ok=True)

    # Lista ordenada das imagens
    imagens = sorted(glob(os.path.join(pasta_entrada, "*.*")))

    for caminho in tqdm(imagens, desc="Processando imagens"):
        nome_arquivo = os.path.basename(caminho)
        img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)  # mantém 0/1 se for máscara

        if img is None:
            print(f"⚠️ Erro ao ler {nome_arquivo}, ignorando.")
            continue

        # Mostra o shape original
        h, w = img.shape[:2]
        # print(f"{nome_arquivo}: {w}x{h}")

        # Calcula o tamanho do quadrado
        tamanho_max = max(h, w)

        # Cria uma nova imagem quadrada preenchida com zeros
        if len(img.shape) == 2:
            # imagem em tons de cinza
            img_quadrada = np.zeros((tamanho_max, tamanho_max), dtype=img.dtype)
        else:
            # imagem colorida
            img_quadrada = np.zeros((tamanho_max, tamanho_max, img.shape[2]), dtype=img.dtype)


        # Calcula as posições de colagem (para centralizar)
        x_offset = (tamanho_max - w) // 2
        y_offset = (tamanho_max - h) // 2

        # Copia a imagem original para o centro da quadrada
        img_quadrada[y_offset:y_offset + h, x_offset:x_offset + w] = img

        # Redimensiona para 512x512
        img_redimensionada = cv2.resize(img_quadrada, (size, size), interpolation=cv2.INTER_NEAREST)

        # virar imagem
        # img_redimensionada = cv2.rotate(img_redimensionada, cv2.ROTATE_90_CLOCKWISE)

        # Salva mantendo o nome original
        cv2.imwrite(os.path.join(pasta_saida, nome_arquivo), img_redimensionada)

        # Testar em uma só imagem
        # break

    print("\nProcesso concluído com sucesso!")