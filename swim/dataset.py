from pathlib import Path
import json

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

import config


class PoreDataset(Dataset):

    def __init__(
        self,
        dataset_root,
        samples,
        target="small",
        normalization_file=None,
        transform=None,
    ):

        self.dataset_root = Path(dataset_root)
        self.samples = list(samples)
        self.target = target.lower()
        self.transform = transform

        # ====================================================
        # TARGET
        # ====================================================

        if self.target not in ("small", "large"):

            raise ValueError(
                f"Target inválido: {self.target}. "
                "Utilize 'small' ou 'large'."
            )

        if self.target == "small":
            self.mask_folder = config.SMALL_MASK_FOLDER

        else:
            self.mask_folder = config.LARGE_MASK_FOLDER

        # ====================================================
        # NORMALIZAÇÃO
        # ====================================================

        if normalization_file is None:

            normalization_file = ( "normalization_stats.json" )

        self.normalization_file = Path(
            normalization_file
        )

        if not self.normalization_file.exists():

            raise FileNotFoundError(
                "Arquivo de normalização não encontrado:\n"
                f"{self.normalization_file}"
            )

        with open(
            self.normalization_file,
            "r",
            encoding="utf-8",
        ) as file:

            normalization = json.load(file)

        self.clip_min = float(
            normalization["clip_min"]
        )

        self.clip_max = float(
            normalization["clip_max"]
        )

        self.normalization_mode = normalization.get(
            "normalization_mode",
            "legacy",
        )

        self.normalization_train_samples = (
            normalization.get(
                "training_samples",
                [],
            )
        )

        if self.clip_max <= self.clip_min:

            raise ValueError(
                "Parâmetros de normalização inválidos:\n"
                f"clip_min = {self.clip_min}\n"
                f"clip_max = {self.clip_max}"
            )

        # ====================================================
        # DATA
        # ====================================================

        self.data = []

        self._build_dataset()

        if len(self.data) == 0:

            raise RuntimeError(
                "Nenhuma imagem encontrada."
            )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    def _build_dataset(self):

        for sample_name in self.samples:

            sample_path = (
                self.dataset_root
                / sample_name
            )

            original_path = (
                sample_path
                / config.ORIGINAL_FOLDER
            )

            mask_path = (
                sample_path
                / self.mask_folder
            )

            # ------------------------------------------------
            # Verificações
            # ------------------------------------------------

            if not sample_path.exists():

                raise FileNotFoundError(
                    "Amostra não encontrada:\n"
                    f"{sample_path}"
                )

            if not original_path.exists():

                raise FileNotFoundError(
                    "Pasta original não encontrada:\n"
                    f"{original_path}"
                )

            if not mask_path.exists():

                raise FileNotFoundError(
                    "Pasta de máscaras não encontrada:\n"
                    f"{mask_path}"
                )

            # ------------------------------------------------
            # Imagens originais
            # ------------------------------------------------

            original_files = sorted(
                [
                    file
                    for file
                    in original_path.iterdir()
                    if file.suffix.lower()
                    in config.IMAGE_EXTENSIONS
                ]
            )

            if len(original_files) == 0:

                raise RuntimeError(
                    "Nenhuma imagem TIFF encontrada em:\n"
                    f"{original_path}"
                )

            # ------------------------------------------------
            # Original -> máscara
            # ------------------------------------------------

            for image_file in original_files:

                corresponding_mask = (
                    mask_path
                    / image_file.name
                )

                if not corresponding_mask.exists():

                    raise FileNotFoundError(
                        "\nMáscara não encontrada.\n"
                        f"Sample: {sample_name}\n"
                        f"Original: {image_file}\n"
                        f"Esperada: {corresponding_mask}"
                    )

                self.data.append(
                    {
                        "sample": sample_name,
                        "image": image_file,
                        "mask": corresponding_mask,
                    }
                )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.data)

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_image(self, image):

        image = image.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Robust clipping
        # ----------------------------------------------------

        image = np.clip(
            image,
            self.clip_min,
            self.clip_max,
        )

        # ----------------------------------------------------
        # [clip_min, clip_max] -> [0, 1]
        # ----------------------------------------------------

        image = (
            image - self.clip_min
        ) / (
            self.clip_max
            - self.clip_min
        )

        return image.astype(
            np.float32
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        item = self.data[index]

        # ====================================================
        # ORIGINAL
        # ====================================================

        with Image.open(
            item["image"]
        ) as img:

            image = np.array(img)

        # ====================================================
        # MASK
        # ====================================================

        with Image.open(
            item["mask"]
        ) as img:

            mask = np.array(img)

        # ====================================================
        # DIMENSÕES
        # ====================================================

        if image.ndim != 2:

            raise ValueError(
                "Imagem original não é grayscale 2D:\n"
                f"{item['image']}\n"
                f"Shape: {image.shape}"
            )

        if mask.ndim != 2:

            raise ValueError(
                "Máscara não é 2D:\n"
                f"{item['mask']}\n"
                f"Shape: {mask.shape}"
            )

        if image.shape != mask.shape:

            raise ValueError(
                "\nImagem e máscara possuem "
                "dimensões diferentes.\n"
                f"Imagem: {item['image']}\n"
                f"Shape: {image.shape}\n"
                f"Máscara: {item['mask']}\n"
                f"Shape: {mask.shape}"
            )

        expected_shape = (
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        )

        if image.shape != expected_shape:

            raise ValueError(
                "\nDimensão inesperada.\n"
                f"Arquivo: {item['image']}\n"
                f"Encontrado: {image.shape}\n"
                f"Esperado: {expected_shape}"
            )

        # ====================================================
        # NORMALIZAÇÃO
        # ====================================================

        image = self._normalize_image(
            image
        )

        # ====================================================
        # MÁSCARA BINÁRIA
        # ====================================================

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ====================================================
        # NUMPY -> TORCH
        # ====================================================

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        # ====================================================
        # GRAYSCALE -> 3 CANAIS
        # ====================================================

        if config.REPLICATE_GRAYSCALE_TO_RGB:

            image = image.repeat(
                3,
                1,
                1,
            )

        # ====================================================
        # TRANSFORM
        # ====================================================

        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask,
            )

        return {
            "image": image,
            "mask": mask,
            "sample": item["sample"],
            "filename": item["image"].name,
        }

    # ========================================================
    # INFO
    # ========================================================

    def print_info(self):

        print("=" * 60)
        print("PORE DATASET")
        print("=" * 60)

        print(
            f"Target:              {self.target}"
        )

        print(
            f"Samples:             {self.samples}"
        )

        print(
            f"Images:              {len(self)}"
        )

        print(
            f"Mask folder:         {self.mask_folder}"
        )

        print(
            f"Normalization mode:  {self.normalization_mode}"
        )

        print(
            f"Clip min:            {self.clip_min:.4f}"
        )

        print(
            f"Clip max:            {self.clip_max:.4f}"
        )

        print(
            "Normalization train: "
            f"{self.normalization_train_samples}"
        )

        print("=" * 60)