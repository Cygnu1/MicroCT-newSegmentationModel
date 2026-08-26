import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ANTES = r"D:\\Alan\\Doutorado\\Rochas\\data\\MC3_1_P4\\Original_512"
PASTA_DEPOIS = r"D:\\Alan\\Doutorado\\Rochas\\data\\MC3_1_P4\\Pos\\Slices_19um_512"

# ------------------------------------------------------------
# ROI CIRCULAR
# ------------------------------------------------------------
# True  -> usa somente a região circular
# False -> usa a imagem inteira

USAR_ROI_CIRCULAR = True

# Raio da ROI em pixels
RAIO_ROI = 255

# ------------------------------------------------------------

NUM_BINS = 256

EXTENSOES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp"
)


# ============================================================
# CRIAR MÁSCARA CIRCULAR
# ============================================================

def criar_roi_circular(img, raio):

    altura, largura = img.shape[:2]

    # Centro da imagem
    centro_x = largura // 2
    centro_y = altura // 2

    # Coordenadas
    y, x = np.ogrid[:altura, :largura]

    # Equação do círculo
    mascara = (
        (x - centro_x) ** 2 +
        (y - centro_y) ** 2
        <= raio ** 2
    )

    return mascara


# ============================================================
# CARREGAR IMAGENS
# ============================================================

def carregar_imagens(pasta):

    arquivos = sorted([
        f for f in os.listdir(pasta)
        if f.lower().endswith(EXTENSOES)
    ])

    arquivos = arquivos[380:3743]

    imagens = []

    for arquivo in arquivos:

        caminho = os.path.join(
            pasta,
            arquivo
        )

        # Preserva a profundidade original
        # Ex.: uint8 ou uint16
        img = cv2.imread(
            caminho,
            cv2.IMREAD_UNCHANGED
        )

        if img is None:
            print(
                f"Não foi possível abrir: {arquivo}"
            )
            continue

        # Caso seja RGB
        if img.ndim == 3:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

        imagens.append(img)

    print("\nPasta:")
    print(pasta)

    print(
        f"Imagens carregadas: {len(imagens)}"
    )

    return imagens


# ============================================================
# EXTRAIR PIXELS
# ============================================================

def extrair_pixels(imagens):

    todos_pixels = []

    for img in imagens:

        if USAR_ROI_CIRCULAR:

            mascara = criar_roi_circular(
                img,
                RAIO_ROI
            )

            # Somente pixels dentro da ROI
            pixels = img[mascara]

        else:

            # Imagem inteira
            pixels = img.ravel()

        todos_pixels.append(
            pixels.astype(np.float64)
        )

    return np.concatenate(todos_pixels)


# ============================================================
# CARREGAR VOLUMES
# ============================================================

imagens_antes = carregar_imagens(
    PASTA_ANTES
)

imagens_depois = carregar_imagens(
    PASTA_DEPOIS
)


# ============================================================
# VERIFICAÇÃO
# ============================================================

if len(imagens_antes) == 0:
    raise ValueError(
        "Nenhuma imagem encontrada na pasta ANTES."
    )

if len(imagens_depois) == 0:
    raise ValueError(
        "Nenhuma imagem encontrada na pasta DEPOIS."
    )


# ============================================================
# INFORMAÇÕES DA ROI
# ============================================================

print("\n" + "=" * 60)
print("CONFIGURAÇÃO DA ANÁLISE")
print("=" * 60)

if USAR_ROI_CIRCULAR:

    print("ROI circular: SIM")
    print(f"Raio: {RAIO_ROI} pixels")

    altura, largura = imagens_antes[0].shape[:2]

    print(
        f"Centro da ROI: "
        f"({largura // 2}, {altura // 2})"
    )

else:

    print("ROI circular: NÃO")
    print("Será utilizada a imagem inteira.")


# ============================================================
# EXTRAIR PIXELS DOS VOLUMES
# ============================================================

pixels_antes = extrair_pixels(
    imagens_antes
)

pixels_depois = extrair_pixels(
    imagens_depois
)


print("\nPixels utilizados:")

print(
    f"Antes : {len(pixels_antes):,}"
)

print(
    f"Depois: {len(pixels_depois):,}"
)


# ============================================================
# DEFINIR INTERVALO COMUM DOS HISTOGRAMAS
# ============================================================

valor_min = min(
    pixels_antes.min(),
    pixels_depois.min()
)

valor_max = max(
    pixels_antes.max(),
    pixels_depois.max()
)

bins = np.linspace(
    valor_min,
    valor_max,
    NUM_BINS + 1
)


# ============================================================
# HISTOGRAMAS
# ============================================================

hist_antes, edges = np.histogram(
    pixels_antes,
    bins=bins
)

hist_depois, _ = np.histogram(
    pixels_depois,
    bins=bins
)


# Centros dos bins

centros = (
    edges[:-1] +
    edges[1:]
) / 2


# ============================================================
# NORMALIZAR HISTOGRAMAS
# ============================================================

hist_antes_prob = (
    hist_antes /
    hist_antes.sum()
)

hist_depois_prob = (
    hist_depois /
    hist_depois.sum()
)


# ============================================================
# ESTATÍSTICAS
# ============================================================

def estatisticas(pixels):

    return {

        "Média":
            np.mean(pixels),

        "Desvio padrão":
            np.std(pixels),

        "Mediana":
            np.median(pixels),

        "Mínimo":
            np.min(pixels),

        "Máximo":
            np.max(pixels),

        "P1":
            np.percentile(pixels, 1),

        "P5":
            np.percentile(pixels, 5),

        "P25":
            np.percentile(pixels, 25),

        "P75":
            np.percentile(pixels, 75),

        "P95":
            np.percentile(pixels, 95),

        "P99":
            np.percentile(pixels, 99),

    }


stats_antes = estatisticas(
    pixels_antes
)

stats_depois = estatisticas(
    pixels_depois
)


# ============================================================
# MOSTRAR ESTATÍSTICAS
# ============================================================

print("\n" + "=" * 60)
print("ESTATÍSTICAS")
print("=" * 60)


print("\nANTES DA INJEÇÃO\n")

for nome, valor in stats_antes.items():

    print(
        f"{nome:20s}: {valor:.4f}"
    )


print("\nDEPOIS DA INJEÇÃO\n")

for nome, valor in stats_depois.items():

    print(
        f"{nome:20s}: {valor:.4f}"
    )


# ============================================================
# DIFERENÇA ENTRE AS ESTATÍSTICAS
# ============================================================

print("\n" + "=" * 60)
print("VARIAÇÃO ANTES -> DEPOIS")
print("=" * 60)


for nome in stats_antes:

    antes = stats_antes[nome]
    depois = stats_depois[nome]

    diferenca = depois - antes

    if antes != 0:

        percentual = (
            diferenca /
            abs(antes)
        ) * 100

    else:

        percentual = np.nan

    print(
        f"{nome:20s}: "
        f"{diferenca:+.4f} "
        f"({percentual:+.2f}%)"
    )


# ============================================================
# DISTÂNCIA WASSERSTEIN
# ============================================================

wasserstein = wasserstein_distance(
    centros,
    centros,
    u_weights=hist_antes_prob,
    v_weights=hist_depois_prob
)


# ============================================================
# DISTÂNCIA JENSEN-SHANNON
# ============================================================

# Pequeno valor para evitar problemas numéricos

epsilon = 1e-12

hist_antes_js = (
    hist_antes_prob +
    epsilon
)

hist_depois_js = (
    hist_depois_prob +
    epsilon
)


hist_antes_js /= hist_antes_js.sum()
hist_depois_js /= hist_depois_js.sum()


js = jensenshannon(
    hist_antes_js,
    hist_depois_js
)


# ============================================================
# DIFERENÇA ABSOLUTA MÉDIA
# ============================================================

diferenca_media_hist = np.mean(
    np.abs(
        hist_antes_prob -
        hist_depois_prob
    )
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("COMPARAÇÃO DOS HISTOGRAMAS")
print("=" * 60)

print(
    f"Distância Wasserstein    : "
    f"{wasserstein:.6f}"
)

print(
    f"Distância Jensen-Shannon : "
    f"{js:.6f}"
)

print(
    f"Diferença absoluta média : "
    f"{diferenca_media_hist:.8f}"
)


# ============================================================
# GRÁFICO DO HISTOGRAMA
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    centros,
    hist_antes_prob,
    label="Antes da injeção",
    linewidth=2
)


plt.plot(
    centros,
    hist_depois_prob,
    label="Depois da injeção",
    linewidth=2
)


plt.xlabel(
    "Intensidade do pixel"
)

plt.ylabel(
    "Probabilidade"
)


if USAR_ROI_CIRCULAR:

    titulo = (
        f"Histogramas - ROI circular "
        f"(raio = {RAIO_ROI} px)"
    )

else:

    titulo = (
        "Histogramas - Imagem inteira"
    )


plt.title(titulo)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# MOSTRAR ROI UTILIZADA
# ============================================================

if USAR_ROI_CIRCULAR:

    img_exemplo = imagens_antes[0]

    mascara = criar_roi_circular(
        img_exemplo,
        RAIO_ROI
    )

    # Criar imagem apenas para visualização
    img_roi = img_exemplo.copy()

    # Pixels externos ficam pretos
    img_roi[~mascara] = 0


    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        img_roi,
        cmap="gray"
    )

    plt.title(
        f"ROI utilizada - Raio = {RAIO_ROI} px"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()