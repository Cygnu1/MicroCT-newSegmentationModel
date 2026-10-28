import random

import torch
from torch.utils.data import Dataset


class PoreTrainAugmentation:
    """
    Data augmentation conservador para micro-CT.

    Geometria:
      - horizontal flip
      - vertical flip
      - rotação em múltiplos de 90 graus

    Intensidade (somente imagem):
      - brilho
      - contraste
      - gamma

    A imagem deve estar normalizada em [0, 1].
    A máscara permanece binária.
    """

    def __init__(
        self,
        horizontal_flip_p=0.5,
        vertical_flip_p=0.5,
        rotation_p=0.75,
        intensity_p=0.5,
        brightness_delta=0.05,
        contrast_range=(0.90, 1.10),
        gamma_range=(0.90, 1.10),
    ):
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotation_p = rotation_p
        self.intensity_p = intensity_p
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range

    def __call__(self, image, mask):

        # ====================================================
        # GEOMETRIC AUGMENTATION
        # image e mask recebem EXATAMENTE a mesma transformação
        # ====================================================

        if random.random() < self.horizontal_flip_p:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if random.random() < self.vertical_flip_p:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        if random.random() < self.rotation_p:
            k = random.choice([1, 2, 3])
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        # ====================================================
        # INTENSITY AUGMENTATION
        # somente a imagem é alterada
        # ====================================================

        if random.random() < self.intensity_p:

            # Contraste
            contrast = random.uniform(
                self.contrast_range[0],
                self.contrast_range[1],
            )

            mean = image.mean()
            image = (
                (image - mean) * contrast
                + mean
            )

            # Brilho
            brightness = random.uniform(
                -self.brightness_delta,
                self.brightness_delta,
            )

            image = image + brightness

            # Mantém o domínio esperado
            image = torch.clamp(
                image,
                0.0,
                1.0,
            )

            # Gamma
            gamma = random.uniform(
                self.gamma_range[0],
                self.gamma_range[1],
            )

            image = torch.pow(
                image,
                gamma,
            )

            image = torch.clamp(
                image,
                0.0,
                1.0,
            )

        return (
            image.contiguous(),
            mask.contiguous(),
        )


class TransformSubset(Dataset):
    """
    Aplica transformação apenas ao Subset fornecido.

    Isso é importante no modo center_block porque train e validation
    compartilham o mesmo PoreDataset base. Assim, augmentation fica
    restrito ao TRAIN e nunca chega à validação.
    """

    def __init__(
        self,
        subset,
        transform=None,
    ):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):

        sample = self.subset[index]

        image = sample["image"]
        mask = sample["mask"]

        if self.transform is not None:
            image, mask = self.transform(
                image,
                mask,
            )

        return {
            "image": image,
            "mask": mask,
            "sample": sample["sample"],
            "filename": sample["filename"],
        }
