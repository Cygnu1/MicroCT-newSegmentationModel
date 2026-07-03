import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob 

def read_image(path: str, in_channels: int) -> np.ndarray:
    if in_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = img[..., None]  # (H,W,1)
    else:
        print("path: ", path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H,W,3)
    return img


def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 127).astype(np.float32)  # robusto p/ 0/255
    return m[..., None]  # (H,W,1)

if __name__ == "__main__":
    # img = 
    path =  "C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\5173\\5173_hr-20260602T122125Z-3-001\\5173_HM\\Original_512"
    # path = 'C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data_HM\\mc3_2_P2\\Original_512\\MC2P2_19um_0039.tiff'

    files = glob(path + "\\*.tiff")

    file = files[0]

    # img = read_image(file,3)
    img1 = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(file, cv2.IMREAD_COLOR)
    print("UNCHANGED:", None if img1 is None else img1.shape, img1.dtype if img1 is not None else None)
    print("GRAYSCALE:", None if img2 is None else img2.shape)
    print("COLOR:", None if img3 is None else img3.shape)
    # img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    msk = read_mask(file)


    print("shape:", img.shape)

    print("Tipo: ", img.dtype)

    print("max: ", np.max(img))
    print("Min: ", np.min(img))

    plt.imshow(img, cmap='gray')
    plt.show()