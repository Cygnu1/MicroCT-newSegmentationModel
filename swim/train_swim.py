from pathlib import Path
import csv
import json
import random
import shutil
import time
from tqdm import tqdm

import numpy as np
import torch

from torch.utils.data import DataLoader

import config

from dataset import PoreDataset

from split_dataset import (
    split_train_validation,
    split_cross_sample,
    print_split_information,
)

from models.swim_segmenter import (
    SwinSegmenter,
)

from losses import (
    BCEDiceLoss,
    TverskyLoss,
)

from metrics import SegmentationMetrics

from augmentations import (
    PoreTrainAugmentation,
    TransformSubset,
)


# ============================================================
# TWO-STAGE FINE-TUNING
# ============================================================

FREEZE_ENCODER_EPOCHS = 100
ENCODER_LEARNING_RATE = 1e-5
DECODER_LEARNING_RATE = config.LEARNING_RATE


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Maior reprodutibilidade.
    # Pode reduzir um pouco a performance.

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def create_experiment_directory():

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if config.VAL_SPLIT_MODE == "cross_sample":
        train_name = "_".join(config.CROSS_TRAIN_SAMPLES)
        val_name = "_".join(config.CROSS_VAL_SAMPLES)
        test_name = "_".join(config.TEST_SAMPLES)

        experiment_name = (
            f"swin_t_{config.TARGET}"
            f"__{config.LOSS_NAME}"
            f"__cross_sample"
            f"__train_{train_name}"
            f"__val_{val_name}"
            f"__test_{test_name}"
            f"__{timestamp}"
        )
    else:
        train_name = "_".join(config.TRAIN_SAMPLES)
        test_name = "_".join(config.TEST_SAMPLES)

        experiment_name = (
            f"swin_t_{config.TARGET}"
            f"__{config.LOSS_NAME}"
            f"__center_block"
            f"__train_{train_name}"
            f"__test_{test_name}"
            f"__{timestamp}"
        )

    output_dir = Path(config.OUTPUT_ROOT) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_experiment_config(
    output_dir,
    split_information,
):

    experiment_config = {

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        "target":
            config.TARGET,

        "train_samples":
            (
                list(config.CROSS_TRAIN_SAMPLES)
                if config.VAL_SPLIT_MODE == "cross_sample"
                else list(config.TRAIN_SAMPLES)
            ),

        "val_samples":
            (
                list(config.CROSS_VAL_SAMPLES)
                if config.VAL_SPLIT_MODE == "cross_sample"
                else list(config.TRAIN_SAMPLES)
            ),

        "test_samples":
            list(config.TEST_SAMPLES),

        "image_height":
            config.IMAGE_HEIGHT,

        "image_width":
            config.IMAGE_WIDTH,

        "validation_fraction":
            config.VAL_FRACTION,

        "validation_split_mode":
            config.VAL_SPLIT_MODE,

        "data_augmentation":
            config.USE_DATA_AUGMENTATION,

        "augmentation_settings": {
            "horizontal_flip_p": config.AUG_HORIZONTAL_FLIP_P,
            "vertical_flip_p": config.AUG_VERTICAL_FLIP_P,
            "rotation_p": config.AUG_ROTATION_P,
            "intensity_p": config.AUG_INTENSITY_P,
            "brightness_delta": config.AUG_BRIGHTNESS_DELTA,
            "contrast_range": list(config.AUG_CONTRAST_RANGE),
            "gamma_range": list(config.AUG_GAMMA_RANGE),
        },

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "model":
            "Swin-T",

        "pretrained":
            True,

        "fine_tuning": {
            "strategy": "freeze_then_unfreeze",
            "freeze_encoder_epochs": FREEZE_ENCODER_EPOCHS,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "decoder_learning_rate": DECODER_LEARNING_RATE,
        },

        "input_channels":
            3,

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        "epochs":
            config.EPOCHS,

        "batch_size":
            config.BATCH_SIZE,

        "learning_rate":
            config.LEARNING_RATE,

        "weight_decay":
            config.WEIGHT_DECAY,

        "bce_weight":
            config.BCE_WEIGHT,

        "dice_weight":
            config.DICE_WEIGHT,

        "threshold":
            config.THRESHOLD,

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        "loss_name":
            config.LOSS_NAME,

        "bce_weight":
            config.BCE_WEIGHT,

        "dice_weight":
            config.DICE_WEIGHT,

        "tversky_alpha":
            config.TVERSKY_ALPHA,

        "tversky_beta":
            config.TVERSKY_BETA,

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        "scheduler":
            "ReduceLROnPlateau",

        "lr_patience":
            config.LR_PATIENCE,

        "lr_factor":
            config.LR_FACTOR,

        "min_lr":
            config.MIN_LR,

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        "early_stopping_patience":
            config.EARLY_STOPPING_PATIENCE,

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        "gradient_clip":
            config.GRADIENT_CLIP,

        # ----------------------------------------------------
        # Seed
        # ----------------------------------------------------

        "seed":
            config.SEED,

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        "split_information":
            split_information,
    }

    output_file = (
        output_dir
        / "experiment_config.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_config,
            file,
            indent=4,
        )


# ============================================================
# COPY NORMALIZATION
# ============================================================

def copy_normalization_file(
    output_dir,
):

    # source = (
    #     Path(config.PROJECT_ROOT)
    #     / "normalization_stats.json"
    # )

    source = (
        Path("normalization_stats.json")
    )

    if not source.exists():

        raise FileNotFoundError(
            "normalization_stats.json "
            "não encontrado."
        )

    destination = (
        output_dir
        / "normalization_stats.json"
    )

    shutil.copy2(
        source,
        destination,
    )


# ============================================================
# CHECK NORMALIZATION
# ============================================================

def validate_normalization():

    # normalization_file = (
    #     Path(config.PROJECT_ROOT)
    #     / "normalization_stats.json"
    # )

    normalization_file = (
        Path("normalization_stats.json")
    )


    with open(
        normalization_file,
        "r",
        encoding="utf-8",
    ) as file:

        normalization = json.load(file)

    normalization_samples = set(
        normalization.get(
            "training_samples",
            [],
        )
    )

    if config.VAL_SPLIT_MODE == "cross_sample":
        training_samples = set(config.CROSS_TRAIN_SAMPLES)
    else:
        training_samples = set(config.TRAIN_SAMPLES)

    if normalization_samples != training_samples:

        raise RuntimeError(
            "\nNORMALIZATION MISMATCH!\n\n"
            "O normalization_stats.json foi "
            "calculado com:\n"
            f"{sorted(normalization_samples)}\n\n"
            "Mas o treinamento atual utiliza:\n"
            f"{sorted(training_samples)}\n\n"
            "Execute novamente "
            "compute_normalization.py."
        )

    print(
        "\n✓ Normalização corresponde "
        "às training samples."
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders():

    if config.VAL_SPLIT_MODE == "center_block":

        full_train_dataset = PoreDataset(
            dataset_root=config.DATASET_ROOT,
            samples=config.TRAIN_SAMPLES,
            target=config.TARGET,
        )

        train_dataset, val_dataset, split_information = (
            split_train_validation(
                full_train_dataset,
                val_fraction=config.VAL_FRACTION,
            )
        )

    elif config.VAL_SPLIT_MODE == "cross_sample":

        train_val_samples = list(
            dict.fromkeys(
                list(config.CROSS_TRAIN_SAMPLES)
                + list(config.CROSS_VAL_SAMPLES)
            )
        )

        full_train_dataset = PoreDataset(
            dataset_root=config.DATASET_ROOT,
            samples=train_val_samples,
            target=config.TARGET,
        )

        train_dataset, val_dataset, split_information = (
            split_cross_sample(
                dataset=full_train_dataset,
                train_samples=config.CROSS_TRAIN_SAMPLES,
                val_samples=config.CROSS_VAL_SAMPLES,
            )
        )

    else:
        raise ValueError(
            f"VAL_SPLIT_MODE inválido: {config.VAL_SPLIT_MODE}"
        )

    # --------------------------------------------------------
    # DATA AUGMENTATION - SOMENTE TRAIN
    # --------------------------------------------------------

    if config.USE_DATA_AUGMENTATION:

        train_transform = PoreTrainAugmentation(
            horizontal_flip_p=config.AUG_HORIZONTAL_FLIP_P,
            vertical_flip_p=config.AUG_VERTICAL_FLIP_P,
            rotation_p=config.AUG_ROTATION_P,
            intensity_p=config.AUG_INTENSITY_P,
            brightness_delta=config.AUG_BRIGHTNESS_DELTA,
            contrast_range=config.AUG_CONTRAST_RANGE,
            gamma_range=config.AUG_GAMMA_RANGE,
        )

        train_dataset = TransformSubset(
            train_dataset,
            transform=train_transform,
        )

        print("\nData augmentation: ENABLED (TRAIN only)")

    else:

        print("\nData augmentation: DISABLED")

    test_dataset = PoreDataset(
        dataset_root=config.DATASET_ROOT,
        samples=config.TEST_SAMPLES,
        target=config.TARGET,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, split_information


# ============================================================
# FINE-TUNING HELPERS
# ============================================================

def get_decoder_parameters(model):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]


def build_frozen_optimizer(model):
    return torch.optim.AdamW(
        get_decoder_parameters(model),
        lr=DECODER_LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )


def build_unfrozen_optimizer(model):
    return torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": get_decoder_parameters(model),
                "lr": DECODER_LEARNING_RATE,
            },
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.MIN_LR,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    metrics = SegmentationMetrics(
        threshold=config.THRESHOLD
    )

    for batch in tqdm(loader):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
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

            loss = criterion(
                logits,
                masks,
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.GRADIENT_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics.update(
            logits.detach(),
            masks,
        )

    average_loss = (
        total_loss
        / total_samples
    )

    metric_results = (
        metrics.compute()
    )

    return (
        average_loss,
        metric_results,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    metrics = SegmentationMetrics(
        threshold=config.THRESHOLD
    )

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        metrics.update(
            logits,
            masks,
        )

    average_loss = (
        total_loss
        / total_samples
    )

    metric_results = (
        metrics.compute()
    )

    return (
        average_loss,
        metric_results,
    )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(
    history,
    output_dir,
):

    output_file = (
        output_dir
        / "training_history.csv"
    )

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_dice",
        "train_iou",
        "train_precision",
        "train_recall",
        "train_specificity",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_specificity",
        "epoch_time_seconds",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# SAVE TEST METRICS
# ============================================================

def save_test_metrics(
    loss,
    metrics,
    output_dir,
):

    output_file = (
        output_dir
        / "test_metrics.csv"
    )

    row = {
        "test_loss": loss,
        "dice": metrics["dice"],
        "iou": metrics["iou"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        writer.writeheader()
        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        config.SEED
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
    )

    print("=" * 80)
    print("SWIN-T PORE SEGMENTATION")
    print("=" * 80)

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"AMP: {use_amp}"
    )

    print(
        f"Target: {config.TARGET}"
    )

    print(
        f"Validation mode: "
        f"{config.VAL_SPLIT_MODE}"
    )

    if config.VAL_SPLIT_MODE == "cross_sample":
        print(
            f"Train samples: "
            f"{config.CROSS_TRAIN_SAMPLES}"
        )
        print(
            f"Validation samples: "
            f"{config.CROSS_VAL_SAMPLES}"
        )
    else:
        print(
            f"Train samples: "
            f"{config.TRAIN_SAMPLES}"
        )

    print(
        f"Test samples: "
        f"{config.TEST_SAMPLES}"
    )

    # ========================================================
    # NORMALIZATION CHECK
    # ========================================================

    validate_normalization()

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = (
        create_experiment_directory()
    )

    print(
        f"\nOutput:\n{output_dir}"
    )

    copy_normalization_file(
        output_dir
    )

    # ========================================================
    # DATA
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader,
        split_information,
    ) = create_dataloaders()

    print_split_information(
        split_information
    )

    print("\nDataset sizes:")

    print(
        f"Train:      "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test:       "
        f"{len(test_loader.dataset)}"
    )

    save_experiment_config(
        output_dir,
        split_information,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading Swin-T...")

    model = SwinSegmenter(
        pretrained=True,
    ).to(device)

    # ============================================================
    # LOSS
    # ============================================================

    if config.LOSS_NAME == "bce_dice":

        criterion = BCEDiceLoss(
            bce_weight=config.BCE_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
        )

    elif config.LOSS_NAME == "tversky":

        criterion = TverskyLoss(
            alpha=config.TVERSKY_ALPHA,
            beta=config.TVERSKY_BETA,
        )

    else:

        raise ValueError(
            f"Loss desconhecida: "
            f"{config.LOSS_NAME}"
        )


    print(
        f"\nLoss: {config.LOSS_NAME}"
    )

    if config.LOSS_NAME == "tversky":

        print(
            f"Tversky alpha: "
            f"{config.TVERSKY_ALPHA}"
        )

        print(
            f"Tversky beta: "
            f"{config.TVERSKY_BETA}"
        )

    # ========================================================
    # TWO-STAGE FINE-TUNING
    # ========================================================

    model.freeze_encoder()

    optimizer = build_frozen_optimizer(model)
    scheduler = build_scheduler(optimizer)

    print(
        f"\nFine-tuning strategy:"
        f"\n  Phase 1: encoder frozen for {FREEZE_ENCODER_EPOCHS} epochs"
        f"\n  Decoder LR: {DECODER_LEARNING_RATE:.2e}"
        f"\n  Phase 2 encoder LR: {ENCODER_LEARNING_RATE:.2e}"
    )

    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_val_dice = -1.0

    epochs_without_improvement = 0

    history = []

    best_model_path = (
        output_dir
        / "best_model.pth"
    )

    start_training_time = (
        time.time()
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        config.EPOCHS + 1,
    ):

        epoch_start = (
            time.time()
        )

        # ----------------------------------------------------
        # Transition to phase 2
        # ----------------------------------------------------

        if epoch == FREEZE_ENCODER_EPOCHS + 1:

            print("\n" + "=" * 80)
            print("UNFREEZING SWIN ENCODER")
            print("=" * 80)

            model.unfreeze_encoder()

            optimizer = build_unfrozen_optimizer(model)
            scheduler = build_scheduler(optimizer)

            # Dá à fase 2 sua própria janela de early stopping.
            epochs_without_improvement = 0

            print(f"Encoder LR: {ENCODER_LEARNING_RATE:.2e}")
            print(f"Decoder LR: {DECODER_LEARNING_RATE:.2e}")

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_metrics,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_metrics,
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["dice"]
        )

        if model.encoder_is_frozen():
            encoder_lr = 0.0
            decoder_lr = optimizer.param_groups[0]["lr"]
            training_phase = "decoder_only"
        else:
            encoder_lr = optimizer.param_groups[0]["lr"]
            decoder_lr = optimizer.param_groups[1]["lr"]
            training_phase = "full_finetune"

        current_lr = decoder_lr

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history_row = {

            "epoch":
                epoch,

            "learning_rate":
                current_lr,

            "train_loss":
                train_loss,

            "train_dice":
                train_metrics["dice"],

            "train_iou":
                train_metrics["iou"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_specificity":
                train_metrics["specificity"],

            "val_loss":
                val_loss,

            "val_dice":
                val_metrics["dice"],

            "val_iou":
                val_metrics["iou"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_specificity":
                val_metrics["specificity"],

            "epoch_time_seconds":
                epoch_time,
        }

        history.append(
            history_row
        )

        # Salva a cada época para não perder histórico
        save_history(
            history,
            output_dir,
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:03d}/"
            f"{config.EPOCHS:03d}"
        )

        print(
            f"Phase: {training_phase} | "
            f"Encoder LR: {encoder_lr:.2e} | "
            f"Decoder LR: {decoder_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        print(
            "TRAIN | "
            f"Loss: {train_loss:.6f} | "
            f"Dice: {train_metrics['dice']:.6f} | "
            f"IoU: {train_metrics['iou']:.6f} | "
            f"Precision: {train_metrics['precision']:.6f} | "
            f"Recall: {train_metrics['recall']:.6f}"
        )

        print(
            "VAL   | "
            f"Loss: {val_loss:.6f} | "
            f"Dice: {val_metrics['dice']:.6f} | "
            f"IoU: {val_metrics['iou']:.6f} | "
            f"Precision: {val_metrics['precision']:.6f} | "
            f"Recall: {val_metrics['recall']:.6f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_metrics["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_metrics["dice"]
            )

            epochs_without_improvement = 0

            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_dice":
                    best_val_dice,

                "target":
                    config.TARGET,

                "train_samples":
                    (
                        list(config.CROSS_TRAIN_SAMPLES)
                        if config.VAL_SPLIT_MODE == "cross_sample"
                        else list(config.TRAIN_SAMPLES)
                    ),

                "val_samples":
                    (
                        list(config.CROSS_VAL_SAMPLES)
                        if config.VAL_SPLIT_MODE == "cross_sample"
                        else list(config.TRAIN_SAMPLES)
                    ),

                "validation_split_mode":
                    config.VAL_SPLIT_MODE,

                "test_samples":
                    list(
                        config.TEST_SAMPLES
                    ),

                "threshold":
                    config.THRESHOLD,

                "training_phase":
                    training_phase,

                "encoder_frozen":
                    model.encoder_is_frozen(),

                "freeze_encoder_epochs":
                    FREEZE_ENCODER_EPOCHS,

                "encoder_learning_rate":
                    ENCODER_LEARNING_RATE,

                "decoder_learning_rate":
                    DECODER_LEARNING_RATE,
            }

            torch.save(
                checkpoint,
                best_model_path,
            )

            print(
                "✓ New best model "
                f"(Dice={best_val_dice:.6f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                "No improvement: "
                f"{epochs_without_improvement}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >=
            config.EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping activated."
            )

            break

    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    total_training_time = (
        time.time()
        - start_training_time
    )

    print("\n" + "=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    print(
        f"\nBest validation Dice: "
        f"{best_val_dice:.6f}"
    )

    print(
        f"Training time: "
        f"{total_training_time / 60:.2f} min"
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\nLoading best model..."
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL TEST")
    print("=" * 80)

    (
        test_loss,
        test_metrics,
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.6f}"
    )

    print(
        f"Dice:        "
        f"{test_metrics['dice']:.6f}"
    )

    print(
        f"IoU:         "
        f"{test_metrics['iou']:.6f}"
    )

    print(
        f"Precision:   "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Recall:      "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"Specificity: "
        f"{test_metrics['specificity']:.6f}"
    )

    print("\nConfusion matrix:")

    print(
        f"TP: {test_metrics['tp']:,}"
    )

    print(
        f"FP: {test_metrics['fp']:,}"
    )

    print(
        f"FN: {test_metrics['fn']:,}"
    )

    print(
        f"TN: {test_metrics['tn']:,}"
    )

    # ========================================================
    # SAVE TEST
    # ========================================================

    save_test_metrics(
        test_loss,
        test_metrics,
        output_dir,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"\nResults saved in:\n"
        f"{output_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()