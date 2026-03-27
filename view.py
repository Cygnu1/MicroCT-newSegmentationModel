import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_image(path: str, in_channels: int) -> np.ndarray:
    if in_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = img[..., None]  # (H,W,1)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H,W,3)
    return img

if __name__ == "__main__":
    # img = 
    file =  "D:\\Alan\\Doutorado\\Rochas\\data\\MC3_3_P4\\fatias_MC33P4_19um_HM\\MC_3_p4_150kV19umSrc72_Stitch (Cropped)0001.tiff"
    # file = 'C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data_HM\\mc3_2_P2\\Original_512\\MC2P2_19um_0039.tiff'

    # img = read_image(file,)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    print("shape:", img.shape)

    print("Tipo: ", img.dtype)

    print("max: ", np.max(img))
    print("Min: ", np.min(img))

    plt.imshow(img, cmap='gray')
    plt.show()