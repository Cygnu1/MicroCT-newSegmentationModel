import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil


def copy_file(source_path, destination_path):
    """
    Copies a file from source_path to destination_path.
    Overwrites the destination if it already exists.
    """

    try:
        # Validate that the source exists and is a file
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Copy the file (preserves file permissions, but not metadata)
        shutil.copy(source_path, destination_path)

        print(f"File copied successfully from '{source_path}' to '{destination_path}'")

    except PermissionError:
        print("Error: Permission denied. Check your file access rights.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if "__main__" == __name__:

    # Pasta de entrada e saída
    pasta_entrada = r"D:\Alan\Doutorado\Rochas\data\MC3_3_P4\Original_HM"   # substitua pelo seu caminho real
    pasta_saida = r"D:\Alan\Doutorado\Rochas\data\MC3_3_P4\Original_HM_bot" 

    # Cria a pasta de saída se não existir
    os.makedirs(pasta_saida, exist_ok=True)

    # Lista ordenada das imagens
    imagens = sorted(glob(os.path.join(pasta_entrada, "*.*")))

    imagens_top = imagens[0:1316]
    imagens_mid = imagens[1317:2632]
    imagens_bot = imagens[2633:3948]


    for caminho in tqdm(imagens_bot, desc="Processando imagens"):
        
        copy_file(caminho, pasta_saida)

        # Testar em uma só imagem
        # break

    print("\nProcesso concluído com sucesso!")