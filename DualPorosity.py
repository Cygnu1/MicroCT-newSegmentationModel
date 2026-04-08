import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Caminhos das pastas
pasta = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\results\roi_p2\resnext101_32x8d_CBAM_512_tversky_HMP2_roi_P2\pred_bin"
pastaIsmael = r"C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HMP2\mc3_2_P2\Segmented_512"

# Parâmetros
altura = 0.8
raio_analise = 0.38


def calcular_porosidade_da_pasta(caminho_pasta, extensao="*.tiff", raio_analise=0.38):
    imagens = sorted(glob(os.path.join(caminho_pasta, extensao)))
    n = len(imagens)

    if n == 0:
        raise ValueError(f"Nenhuma imagem encontrada em: {caminho_pasta}")

    inicio = int(0.1 * n)
    fim = int(0.9 * n)

    print(f"\nPasta: {caminho_pasta}")
    print(f"Total de imagens: {n}")
    print(f"Começa na fatia {inicio} e vai até a fatia {fim}")

    imagens_selecionadas = imagens[inicio:fim]
    print(f"Estão sendo usadas {len(imagens_selecionadas)} fatias")

    porosidades = []
    indices = []

    total_pixels_poro = 0
    total_pixels_geral = 0

    for i, caminho in tqdm(enumerate(imagens_selecionadas, start=inicio), total=len(imagens_selecionadas)):
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Erro ao ler: {caminho}")
            continue

        # Garante que é binária (0 ou 1)
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

        h, w = img.shape
        centro = (w // 2, h // 2)
        raio = int(raio_analise * min(w, h))

        # Máscara circular
        mascara_circular = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mascara_circular, centro, raio, 1, -1)

        # Pixels dentro do círculo
        dentro_circulo = img[mascara_circular == 1]

        # Aqui estou mantendo sua lógica:
        # porosidade = fração de pixels com valor 1
        poros = np.sum(dentro_circulo == 1)
        total = dentro_circulo.size
        porosidade = poros / total

        porosidades.append(porosidade)
        indices.append(i)

        total_pixels_poro += poros
        total_pixels_geral += total

    media_porosidade = np.mean(porosidades)
    porosidade_total = total_pixels_poro / total_pixels_geral

    print(f"Porosidade média das imagens: {media_porosidade:.4f}")
    print(f"Porosidade total (global): {porosidade_total:.4f}")

    return indices, porosidades, media_porosidade, porosidade_total


# Calcula nas duas pastas
indices1, porosidades1, media1, total1 = calcular_porosidade_da_pasta(
    pasta, extensao="*.tiff", raio_analise=raio_analise
)

indices2, porosidades2, media2, total2 = calcular_porosidade_da_pasta(
    pastaIsmael, extensao="*.tiff", raio_analise=raio_analise
)

# --- Gráfico comparando as duas pastas ---
plt.figure(figsize=(8, 10))

plt.scatter(porosidades1, range(len(porosidades1)), s=20, label='ML')
plt.scatter(porosidades2, range(len(porosidades2)), s=20, label='Ismael')

# plt.axvline(media1, linestyle='--', label=f'Média pasta = {media1:.4f}')
# plt.axvline(total1, linestyle='-.', label=f'Total pasta = {total1:.4f}')

# plt.axvline(media2, linestyle='--', label=f'Média pastaIsmael = {media2:.4f}')
# plt.axvline(total2, linestyle='-.', label=f'Total pastaIsmael = {total2:.4f}')

plt.gca().invert_yaxis()
plt.title('Porosidade por Imagem (80% centrais)')
plt.xlabel('Porosidade')
plt.ylabel('Índice da imagem')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()