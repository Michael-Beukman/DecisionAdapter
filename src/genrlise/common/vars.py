import os


SAVE_PDF = True
SAVE_PNG = False
SAVE_EPS = False
SAVE_SVG = False
SAVE_JPG = True

FIG_DPI = 400

LOG_WANDB_TABLE = False
# Where all of the output is stored
ARTIFACTS_DIR       = 'artifacts'
RESULTS_DIR         = 'artifacts/results'
PROCESSED_DIR       = 'artifacts/processed'
MODELS_DIR          = 'artifacts/models'
TENSORBOARD_DIR     = 'artifacts/tensorboard'
CONFIG_DIR          = 'artifacts/config'
SLURM_DIR           = 'artifacts/slurms'
SLURM_LOG_DIR       = 'artifacts/logs/slurms'
LOG_DIR             = 'artifacts/logs'

# Create directories
CHECK_FIRST = True
if CHECK_FIRST:
    for a in [RESULTS_DIR, MODELS_DIR, CONFIG_DIR, SLURM_DIR, SLURM_LOG_DIR, TENSORBOARD_DIR]: os.makedirs(a, exist_ok=True)

PROJECT_NAME = 'DynamicsGeneralisation'

def ROOT_DIR(is_local: bool):
    if is_local: 
        return '/path/to/this/repo'
    return '/path/to/this/repo'

def NUM_CORES(is_local: bool):
    return 2 if is_local else 10

NUM_SEEDS = 30

CONDA_ENV_NAME = 'DA'