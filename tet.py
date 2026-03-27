import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# caminhos
img_path = "D:\\Alan\\Doutorado\\Rochas\\data\\MC3_3_P4\\fatias_MC33P4_19um\\MC_3_p4_150kV19umSrc72_Stitch (Cropped)0001.tiff"
ref_path = "C:\\Users\\alanaraujo\\Downloads\\MC3_3_P4_M_19um150kVSrc70_Stitch.tiff"

# leitura das imagens
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)

# histogram matching
matched = match_histograms(img, ref, channel_axis=None)

# flatten para calcular histogramas
img_flat = img.flatten()
ref_flat = ref.flatten()
matched_flat = matched.flatten()

# plot
plt.figure(figsize=(14,6))

# histograma original
plt.subplot(1,3,1)
plt.plot(img)
plt.title("Histograma - Original")

# histograma referência
plt.subplot(1,3,2)
plt.plot(ref)
plt.title("Histograma - Referência")

# histograma depois do matching
plt.subplot(1,3,3)
plt.plot(matched)
plt.title("Histograma - Após Matching")
plt.tight_layout()
plt.show()

# plt.figure(figsize=(8,6))

# plt.hist(img_flat, bins=256, alpha=0.5, label="Original")
# plt.hist(ref_flat, bins=256, alpha=0.5, label="Referência")
# plt.hist(matched_flat, bins=256, alpha=0.5, label="Após Matching")

# plt.legend()
# plt.title("Comparação dos Histogramas")
# plt.xlabel("Intensidade")
# plt.ylabel("Frequência")

# plt.show()

