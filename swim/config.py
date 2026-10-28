from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent

# pasta DL02
DATASET_ROOT = Path("/home/ari/Alan_araujo/Rochas/data/data")
# OUTPUT_ROOT = Path("/home/ari/Alan_araujo/Rochas/codes/segmentation/results")
OUTPUT_ROOT = Path("results")

# pasta maquina 17153
# DATASET_ROOT = Path(r'C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\data_HM')
# OUTPUT_ROOT = Path(r'C:\Users\alanaraujo\Documents\tecgraf\Doutorado\Rocha\Segmentation\results')


# ============================================================
# DATASET FOLDERS
# ============================================================

ORIGINAL_FOLDER = "Original_512"

LARGE_MASK_FOLDER = "segmented_512_large"
SMALL_MASK_FOLDER = "segmented_512_small"


# ============================================================
# SAMPLES
# ============================================================

# Amostras utilizadas para treinamento + validação
TRAIN_SAMPLES = [
    "mc3_2_P4",
    "mc3_2_P2",
]

# Amostras utilizadas exclusivamente para teste
TEST_SAMPLES = [
    "c2d",
]


# ============================================================
# SEGMENTATION TARGET
# ============================================================

# Opções:
# "small"
# "large"

TARGET = "small"


# ============================================================
# IMAGE
# ============================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Swin-T pré-treinado espera 3 canais.
# A imagem grayscale será repetida nos três canais.
REPLICATE_GRAYSCALE_TO_RGB = True


# Extensões aceitas
IMAGE_EXTENSIONS = (
    ".tif",
    ".tiff",
)


# ============================================================
# RADIOMETRIC NORMALIZATION EXPERIMENT
# ============================================================
# Choose manually before running compute_normalization.py:
#   "baseline" -> current pipeline: [0, P99.5] -> [0, 1]
#   "p1_p99"   -> [P1, P99] -> [0, 1]
#   "p2_p98"   -> [P2, P98] -> [0, 1]
NORMALIZATION_MODE = "p1_p99"

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4



EARLY_STOPPING_PATIENCE = 12

LR_PATIENCE = 5
LR_FACTOR = 0.5

MIN_LR = 1e-7

GRADIENT_CLIP = 1.0

THRESHOLD = 0.5

# ============================================================
# LOSS
# ============================================================

LOSS_NAME = "tversky"
TVERSKY_ALPHA = 3.0
TVERSKY_BETA = 1.0

BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# ============================================================
# VALIDATION
# ============================================================

# Fração das imagens das training samples reservada
# para validação.
VAL_FRACTION = 0.10

# Modos:
# "center_block" = bloco central das training samples
# "cross_sample" = amostras inteiras separadas para train/validation
VAL_SPLIT_MODE = "cross_sample"

# Diagnóstico cross-rock atual:
# TRAIN = mc3_2_P2 | VAL = mc3_2_P4 | TEST = c2d
CROSS_TRAIN_SAMPLES = [
    "mc3_2_P4",
]

CROSS_VAL_SAMPLES = [
    "mc3_2_P2",
]


# ============================================================
# DATA AUGMENTATION
# ============================================================

# Aplicado SOMENTE ao conjunto de treinamento.
USE_DATA_AUGMENTATION = True

# Transformações geométricas
AUG_HORIZONTAL_FLIP_P = 0.0
AUG_VERTICAL_FLIP_P = 0.0
AUG_ROTATION_P = 0.0

# Transformações suaves de intensidade
AUG_INTENSITY_P = 0.5
AUG_BRIGHTNESS_DELTA = 0.10
AUG_CONTRAST_RANGE = (0.80, 1.20)
AUG_GAMMA_RANGE = (0.90, 1.10)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 4
NUM_WORKERS = 4
PIN_MEMORY = True