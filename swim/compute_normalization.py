from pathlib import Path
import json

import numpy as np
from PIL import Image

import config


# ============================================================
# CONFIGURAÇÕES DA NORMALIZAÇÃO
# ============================================================

MIN_VALID_VALUE = 0
OUTPUT_FILE = "normalization_stats.json"

NORMALIZATION_MODES = {
    "baseline": {"low_percentile": None, "high_percentile": 99.5, "force_low_zero": True},
    "p1_p99":   {"low_percentile": 1.0,  "high_percentile": 99.0, "force_low_zero": False},
    "p2_p98":   {"low_percentile": 2.0,  "high_percentile": 98.0, "force_low_zero": False},
}

def get_normalization_mode():
    mode = getattr(config, "NORMALIZATION_MODE", "baseline").lower()
    if mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"NORMALIZATION_MODE inválido: {mode}. "
            f"Use: {list(NORMALIZATION_MODES.keys())}"
        )
    return mode

def get_mode_settings():
    return NORMALIZATION_MODES[get_normalization_mode()]

def get_normalization_samples():
    """Retorna somente as amostras efetivamente usadas no treinamento."""
    if config.VAL_SPLIT_MODE == "cross_sample":
        return list(config.CROSS_TRAIN_SAMPLES)
    return list(config.TRAIN_SAMPLES)


# ============================================================
# FUNÇÕES
# ============================================================

def get_tiff_files(folder):
    """
    Retorna arquivos TIFF ordenados.
    """

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.suffix.lower()
            in config.IMAGE_EXTENSIONS
        ]
    )


def load_image(path):
    """
    Carrega TIFF preservando seus valores originais.
    """

    with Image.open(path) as img:
        image = np.array(img)

    return image


# ============================================================
# PRIMEIRA PASSAGEM
# ============================================================

def inspect_training_data():
    """
    Faz uma primeira inspeção das imagens utilizadas
    no treinamento.
    """

    print("=" * 70)
    print("COMPUTE NORMALIZATION")
    print("=" * 70)

    print("\nTraining samples:")

    for sample in get_normalization_samples():
        print(f"  - {sample}")

    print(
        f"\nIgnoring pixels <= {MIN_VALID_VALUE}"
    )

    settings = get_mode_settings()
    print(f"Normalization mode: {get_normalization_mode()}")
    if settings["force_low_zero"]:
        print(f"Range: 0 - P{settings['high_percentile']}")
    else:
        print(
            f"Percentiles: P{settings['low_percentile']} - "
            f"P{settings['high_percentile']}"
        )

    total_images = 0
    total_pixels = 0
    valid_pixels = 0

    global_min = np.inf
    global_max = -np.inf

    files_by_sample = {}

    # --------------------------------------------------------
    # Percorre as amostras de treinamento
    # --------------------------------------------------------

    for sample_name in get_normalization_samples():

        original_folder = (
            config.DATASET_ROOT
            / sample_name
            / config.ORIGINAL_FOLDER
        )

        if not original_folder.exists():

            raise FileNotFoundError(
                f"Pasta não encontrada:\n"
                f"{original_folder}"
            )

        files = get_tiff_files(
            original_folder
        )

        if len(files) == 0:

            raise RuntimeError(
                f"Nenhum TIFF encontrado em:\n"
                f"{original_folder}"
            )

        files_by_sample[sample_name] = files

        sample_total = 0
        sample_valid = 0

        print("\n" + "-" * 70)
        print(f"Sample: {sample_name}")
        print(f"Images: {len(files)}")

        for image_file in files:

            image = load_image(
                image_file
            )

            if image.ndim != 2:

                raise ValueError(
                    f"Imagem não é grayscale 2D:\n"
                    f"{image_file}\n"
                    f"Shape: {image.shape}"
                )

            sample_total += image.size

            valid_mask = (
                image > MIN_VALID_VALUE
            )

            n_valid = np.count_nonzero(
                valid_mask
            )

            sample_valid += n_valid

            if n_valid > 0:

                valid_values = image[
                    valid_mask
                ]

                global_min = min(
                    global_min,
                    float(valid_values.min()),
                )

                global_max = max(
                    global_max,
                    float(valid_values.max()),
                )

        total_images += len(files)

        total_pixels += sample_total
        valid_pixels += sample_valid

        valid_percentage = (
            100.0
            * sample_valid
            / sample_total
        )

        print(
            f"Valid pixels: "
            f"{sample_valid:,}"
        )

        print(
            f"Valid fraction: "
            f"{valid_percentage:.2f}%"
        )

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Images:       {total_images:,}"
    )

    print(
        f"Total pixels: {total_pixels:,}"
    )

    print(
        f"Valid pixels: {valid_pixels:,}"
    )

    print(
        f"Valid ratio:  "
        f"{100 * valid_pixels / total_pixels:.2f}%"
    )

    print(
        f"Valid min:    {global_min:.4f}"
    )

    print(
        f"Valid max:    {global_max:.4f}"
    )

    return (
        files_by_sample,
        valid_pixels,
    )


# ============================================================
# CÁLCULO DOS PERCENTIS
# ============================================================

def compute_percentiles(
    files_by_sample,
    total_valid_pixels,
):
    """
    Calcula os percentis globais considerando somente
    pixels válidos das amostras de treinamento.

    Para evitar uma lista Python gigantesca, os valores
    são armazenados diretamente em um array NumPy.
    """

    print("\n" + "=" * 70)
    print("COLLECTING VALID PIXELS")
    print("=" * 70)

    # uint16 -> 2 bytes por pixel
    memory_gb = (
        total_valid_pixels
        * np.dtype(np.uint16).itemsize
        / (1024 ** 3)
    )

    print(
        f"\nEstimated RAM for pixel array: "
        f"{memory_gb:.2f} GB"
    )

    all_pixels = np.empty(
        total_valid_pixels,
        dtype=np.uint16,
    )

    current_position = 0

    total_images = sum(
        len(files)
        for files
        in files_by_sample.values()
    )

    processed_images = 0

    # --------------------------------------------------------
    # Preenche array
    # --------------------------------------------------------

    for sample_name, files in (
        files_by_sample.items()
    ):

        print(
            f"\nCollecting: {sample_name}"
        )

        for image_file in files:

            image = load_image(
                image_file
            )

            valid_values = image[
                image > MIN_VALID_VALUE
            ]

            n_values = (
                valid_values.size
            )

            all_pixels[
                current_position:
                current_position + n_values
            ] = valid_values

            current_position += n_values

            processed_images += 1

            if (
                processed_images % 500 == 0
                or
                processed_images
                == total_images
            ):

                print(
                    f"Processed: "
                    f"{processed_images}/"
                    f"{total_images}"
                )

    # --------------------------------------------------------
    # Segurança
    # --------------------------------------------------------

    if current_position != total_valid_pixels:

        raise RuntimeError(
            "Quantidade de pixels coletados "
            "não corresponde ao esperado."
        )

    # --------------------------------------------------------
    # Percentis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALCULATING PERCENTILES")
    print("=" * 70)

    percentiles_to_compute = [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        50.0,
        95.0,
        98.0,
        99.0,
        99.5,
        99.9,
    ]

    values = np.percentile(
        all_pixels,
        percentiles_to_compute,
    )

    percentile_results = {}

    for percentile, value in zip(
        percentiles_to_compute,
        values,
    ):

        percentile_results[
            str(percentile)
        ] = float(value)

        print(
            f"P{percentile:<5}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Mean / std dos pixels válidos
    # --------------------------------------------------------

    mean = float(
        np.mean(
            all_pixels,
            dtype=np.float64,
        )
    )

    std = float(
        np.std(
            all_pixels,
            dtype=np.float64,
        )
    )

    print("\nValid pixels:")

    print(
        f"Mean: {mean:.4f}"
    )

    print(
        f"Std:  {std:.4f}"
    )

    return (
        percentile_results,
        mean,
        std,
    )


# ============================================================
# SALVAR RESULTADOS
# ============================================================

def save_results(
    percentiles,
    mean,
    std,
):
    """
    Salva os parâmetros de normalização.
    """

    mode = get_normalization_mode()
    settings = get_mode_settings()

    if settings["force_low_zero"]:
        low_value = 0.0
        low_percentile = None
    else:
        low_percentile = settings["low_percentile"]
        low_value = percentiles[str(float(low_percentile))]

    high_percentile = settings["high_percentile"]
    high_value = percentiles[str(float(high_percentile))]

    results = {

        "training_samples":
            get_normalization_samples(),

        "ignore_pixels_leq":
            MIN_VALID_VALUE,

        "normalization_mode":
            mode,

        "low_percentile":
            low_percentile,

        "high_percentile":
            high_percentile,

        "clip_min":
            low_value,

        "clip_max":
            high_value,

        "valid_pixel_mean":
            mean,

        "valid_pixel_std":
            std,

        "percentiles":
            percentiles,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("NORMALIZATION PARAMETERS")
    print("=" * 70)

    print(f"\nMode: {mode}")
    print(
        f"\nClip min: {low_value:.4f}"
    )

    print(
        f"Clip max: {high_value:.4f}"
    )

    print(
        "\nNormalization:"
    )

    print(
        "x = clip(x, clip_min, clip_max)"
    )

    print(
        "x = (x - clip_min) / "
        "(clip_max - clip_min)"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        files_by_sample,
        total_valid_pixels,
    ) = inspect_training_data()

    (
        percentiles,
        mean,
        std,
    ) = compute_percentiles(
        files_by_sample,
        total_valid_pixels,
    )

    save_results(
        percentiles,
        mean,
        std,
    )


if __name__ == "__main__":
    main()