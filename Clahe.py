import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

def applyClahe(image):
    """ Applying clahe to the image """
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(3,3))
    # final_img = clahe.apply(image) + 50
    final_img = clahe.apply(image)
    return final_img


def create_dir(path):
    """ Creating a directory """
    if not os.path.exists(path):
        os.makedirs(path)

if "__main__" == __name__:

    # Tamanho para redimensionar para quadrado
    size = 512

    # Pasta de entrada e saída
    pasta_entrada = "C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\c2d\\Original_512"   # substitua pelo seu caminho real
    pasta_saida = "C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data_clahe\\c2d\\Original_512"

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

        # print("img max: ", np.max(img))
        # print("img min: ", np.min(img))

        # # Normalizar
        # img = img.astype(np.float32)
        # img = (img - img.min()) / (img.max() - img.min())

        # CLAHE
        clahe = applyClahe(img)
        # print("clahe max: ", np.max(clahe))
        # print("clahe min: ", np.min(clahe))

        # plt.imshow(img, cmap='gray')
        # plt.show()

        # plt.imshow(clahe, cmap='gray')
        # plt.show()

        # Salva mantendo o nome original
        cv2.imwrite(os.path.join(pasta_saida, nome_arquivo), clahe)

        # Testar em uma só imagem
        # break

    print("\nProcesso concluído com sucesso!")
        
