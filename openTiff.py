import tifffile as tiff
import os
from tqdm import tqdm


if "__main__" == __name__:

    # Caminho do arquivo TIFF
    input_file = "C:\\Users\\alanaraujo\\Downloads\\MC3_3_P4_M_19um150kVSrc70_Stitch.tiff"

    # Pasta de saída
    output_folder = "C:\\Users\\alanaraujo\\Documents\\tecgraf\\Doutorado\\Rocha\\Segmentation\\data\\mc3_3_P4"
    os.makedirs(output_folder, exist_ok=True)

    # Lê o TIFF (3D: slices empilhadas)
    image_stack = tiff.imread(input_file)

    print("shape: ",image_stack.shape)

    # Salva cada slice separadamente
    for i, slice_img in tqdm(enumerate(image_stack), desc="Lendo arquivos"):
        output_path = os.path.join(output_folder, f"MC3_3_P4_19um_{i:04d}.tiff")
        tiff.imwrite(output_path, slice_img)
        # break

    print("Slices extraídas com sucesso!")