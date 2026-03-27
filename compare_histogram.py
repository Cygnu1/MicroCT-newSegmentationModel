import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

if __name__ == "__main__":
    # caminhos
    img_path = "C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\c2d\\Original_512\\plug_C2D_0001.tiff"
    ref_path = "C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\mc3_2_P2\\Original_512\\MC2P2_19um_0001.tiff"

    # leitura
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)

    # histogram matching
    matched = match_histograms(img, ref, channel_axis=None)

    # converter para 1D
    img_flat = img.flatten()
    ref_flat = ref.flatten()
    matched_flat = matched.flatten()

    # número de bins (bom para imagens médicas)
    bins = 512

    # ---------- HISTOGRAMAS ----------
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.hist(img_flat, bins=bins, density=True, alpha=0.5, label="Original")
    plt.hist(ref_flat, bins=bins, density=True, alpha=0.5, label="Referência")
    plt.hist(matched_flat, bins=bins, density=True, alpha=0.5, label="Após Matching")

    plt.xlabel("Intensidade do Pixel")
    plt.ylabel("Densidade de Probabilidade")
    plt.title("Distribuição de Intensidade (Histograma)")
    plt.legend()

    # ---------- CDF ----------
    plt.subplot(1,2,2)

    def compute_cdf(data, bins):
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        return bin_edges[:-1], cdf

    x_img, cdf_img = compute_cdf(img_flat, bins)
    x_ref, cdf_ref = compute_cdf(ref_flat, bins)
    x_match, cdf_match = compute_cdf(matched_flat, bins)

    plt.plot(x_img, cdf_img, label="Original")
    plt.plot(x_ref, cdf_ref, label="Referência")
    plt.plot(x_match, cdf_match, label="Após Matching")

    plt.xlabel("Intensidade do Pixel")
    plt.ylabel("Probabilidade Acumulada (CDF)")
    plt.title("Função de Distribuição Cumulativa")
    plt.legend()

    plt.tight_layout()
    plt.show()