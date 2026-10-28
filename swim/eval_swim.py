from pathlib import Path
import json
import time

import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader

import config

from dataset import PoreDataset
from models.swim_segmenter import SwinSegmenter


# ============================================================
# CONFIGURAÇÃO DA INFERÊNCIA
# ============================================================

# Coloque aqui a pasta do experimento que deseja avaliar.

EXPERIMENT_DIR = Path(
    r"/home/ari/Alan_araujo/Rochas/codes/segmentation/swim/results/swin_t_small__tversky__cross_sample__train_mc3_2_P4__val_mc3_2_P2__test_c2d__20261028_100903"
)


# Threshold usado para transformar probabilidade em máscara.
#
# Inicialmente mantemos 0.5 para ser consistente
# com as métricas calculadas no treinamento.

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_amp = (
    device.type == "cuda"
)


# ============================================================
# CHECK FILES
# ============================================================

checkpoint_path = (
    EXPERIMENT_DIR
    / "best_model.pth"
)

normalization_path = (
    EXPERIMENT_DIR
    / "normalization_stats.json"
)

experiment_config_path = (
    EXPERIMENT_DIR
    / "experiment_config.json"
)


if not checkpoint_path.exists():

    raise FileNotFoundError(
        f"Checkpoint não encontrado:\n"
        f"{checkpoint_path}"
    )


if not normalization_path.exists():

    raise FileNotFoundError(
        f"Normalização não encontrada:\n"
        f"{normalization_path}"
    )


if not experiment_config_path.exists():

    raise FileNotFoundError(
        f"Configuração do experimento "
        f"não encontrada:\n"
        f"{experiment_config_path}"
    )


# ============================================================
# LOAD EXPERIMENT CONFIG
# ============================================================

with open(
    experiment_config_path,
    "r",
    encoding="utf-8",
) as file:

    experiment_info = json.load(file)


target = experiment_info["target"]

test_samples = experiment_info[
    "test_samples"
]


print("=" * 80)
print("SWIN-T - PREDICTION")
print("=" * 80)

print(f"\nDevice: {device}")

if torch.cuda.is_available():

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


print(
    f"\nExperiment:\n"
    f"{EXPERIMENT_DIR}"
)

print(
    f"\nTarget: {target}"
)

print(
    f"Test samples: {test_samples}"
)

print(
    f"Threshold: {THRESHOLD}"
)


# ============================================================
# DATASET
# ============================================================

print("\nLoading test dataset...")


test_dataset = PoreDataset(
    dataset_root=config.DATASET_ROOT,
    samples=test_samples,
    target=target,

    # IMPORTANTE:
    # usa exatamente a normalização
    # que foi salva no treinamento.
    normalization_file=normalization_path,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
)


print(
    f"Test images: "
    f"{len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading model...")


# Não precisamos baixar/carregar novamente
# pesos ImageNet porque o checkpoint contém
# todos os pesos treinados.

model = SwinSegmenter(
    pretrained=False,
).to(device)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    f"Checkpoint epoch: "
    f"{checkpoint.get('epoch', 'unknown')}"
)

print(
    f"Best validation Dice: "
    f"{checkpoint.get('best_val_dice', 'unknown')}"
)


# ============================================================
# OUTPUT
# ============================================================

prediction_root = (
    EXPERIMENT_DIR
    / "predictions"
)


prediction_root.mkdir(
    parents=True,
    exist_ok=True,
)


for sample in test_samples:

    sample_output = (
        prediction_root
        / sample
    )

    sample_output.mkdir(
        parents=True,
        exist_ok=True,
    )


print(
    f"\nPredictions will be saved in:\n"
    f"{prediction_root}"
)


# ============================================================
# INFERENCE
# ============================================================

print("\n" + "=" * 80)
print("INFERENCE")
print("=" * 80)


total_images = 0
total_predicted_pixels = 0
total_pixels = 0

start_time = time.time()


with torch.inference_mode():

    for batch_index, batch in enumerate(
        test_loader
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(images)

        # ----------------------------------------------------
        # Logits -> probabilities
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        # ----------------------------------------------------
        # Probabilities -> binary masks
        # ----------------------------------------------------

        predictions = (
            probabilities >= THRESHOLD
        )

        # CPU
        predictions = (
            predictions
            .squeeze(1)
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        sample_names = batch["sample"]
        filenames = batch["filename"]

        # ----------------------------------------------------
        # Save each image
        # ----------------------------------------------------

        for i in range(
            predictions.shape[0]
        ):

            prediction = predictions[i]

            sample_name = sample_names[i]

            filename = filenames[i]

            # --------------------------------------------
            # Binary:
            #
            # False -> 0
            # True  -> 255
            # --------------------------------------------

            output_mask = (
                prediction.astype(
                    np.uint8
                )
                * 255
            )

            output_path = (
                prediction_root
                / sample_name
                / filename
            )

            Image.fromarray(
                output_mask
            ).save(
                output_path
            )

            # --------------------------------------------
            # Statistics
            # --------------------------------------------

            total_predicted_pixels += int(
                prediction.sum()
            )

            total_pixels += (
                prediction.size
            )

            total_images += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_index + 1) % 10 == 0
            or
            (batch_index + 1)
            == len(test_loader)
        ):

            print(
                f"Batch "
                f"{batch_index + 1:4d}/"
                f"{len(test_loader):4d} | "
                f"Images saved: "
                f"{total_images}"
            )


# ============================================================
# FINAL STATISTICS
# ============================================================

elapsed_time = (
    time.time()
    - start_time
)


predicted_fraction = (
    total_predicted_pixels
    / total_pixels
    if total_pixels > 0
    else 0.0
)


print("\n" + "=" * 80)
print("INFERENCE FINISHED")
print("=" * 80)


print(
    f"\nImages processed: "
    f"{total_images}"
)

print(
    f"Predicted pore pixels: "
    f"{total_predicted_pixels:,}"
)

print(
    f"Total pixels: "
    f"{total_pixels:,}"
)

print(
    f"Predicted pore fraction: "
    f"{predicted_fraction * 100:.6f}%"
)

print(
    f"\nTotal time: "
    f"{elapsed_time:.2f} s"
)

if total_images > 0:

    print(
        f"Time/image: "
        f"{elapsed_time / total_images:.4f} s"
    )


print(
    f"\nPredictions saved in:\n"
    f"{prediction_root}"
)

print(
    "\n✓ Prediction completed."
)