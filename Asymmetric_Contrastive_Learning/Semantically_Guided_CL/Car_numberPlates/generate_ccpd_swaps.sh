#!/bin/bash

#==============================================================================
# CCPD Plate-Swapping Dataset Generation
# Submit this from the project root, just outside dataset_formation:
#   mkdir -p logs
#   sbatch generate_ccpd_swaps.sh
#==============================================================================

#SBATCH --job-name=ccpd-swap

#SBATCH --output=logs/ccpd-swap-%j.out
#SBATCH --error=logs/ccpd-swap-%j.err

# Choose a CPU partition on your cluster if available.
#SBATCH --partition=cpu
#SBATCH --qos=cpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# This job does not use GPUs. OpenCV seamlessClone runs on CPU here.
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

# Increase for full CCPD runs if your storage is slower or the node is busy.
#SBATCH --time=12:00:00

# Mailing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tusharyadav3897@gmail.com

set -euo pipefail

#==============================================================================
# Environment
#==============================================================================

export PYTHONUNBUFFERED=1

# Avoid oversubscribing CPU threads inside each worker process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#==============================================================================
# Activate Environment
#==============================================================================

eval "$(mamba shell hook --shell bash)"
mamba activate ml

#==============================================================================
# Directories
#==============================================================================

# Submit from the directory that contains dataset_formation/.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Set these before submitting, or edit them here.
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-$HOME/ChineseCarParkingDataset2019}"
OUTPUT_DATASET_ROOT="${OUTPUT_DATASET_ROOT:-$HOME/ChineseCarParkingDataset2019_plate_swapped}"

cd "$PROJECT_DIR"

mkdir -p logs

#==============================================================================
# Generation Parameters
#==============================================================================

SEED="${SEED:-69}"
WORKERS="${WORKERS:-$SLURM_CPUS_PER_TASK}"
JPEG_QUALITY="${JPEG_QUALITY:-100}"

# Leave LIMIT empty for the full dataset. Example for testing:
#   LIMIT=1000 sbatch generate_ccpd_swaps.sh
LIMIT="${LIMIT:-}"

# Set OVERWRITE=1 to regenerate existing outputs.
OVERWRITE="${OVERWRITE:-0}"

#==============================================================================
# Information
#==============================================================================

echo "========================================================"
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $(hostname)"
echo "Date            : $(date)"
echo "Project Dir     : $PROJECT_DIR"
echo "Source Dataset  : $SOURCE_DATASET_ROOT"
echo "Output Dataset  : $OUTPUT_DATASET_ROOT"
echo "Workers         : $WORKERS"
echo "Seed            : $SEED"
echo "JPEG Quality    : $JPEG_QUALITY"
echo "Limit           : ${LIMIT:-full dataset}"
echo "Overwrite       : $OVERWRITE"
echo "========================================================"

python - <<'PY'
import cv2
import numpy as np
import tqdm
print("OpenCV:", cv2.__version__)
print("NumPy :", np.__version__)
print("tqdm  :", tqdm.__version__)
PY

#==============================================================================
# Launch
#==============================================================================

ARGS=(
    --dataset-root "$SOURCE_DATASET_ROOT"
    --output-root "$OUTPUT_DATASET_ROOT"
    --workers "$WORKERS"
    --seed "$SEED"
    --jpeg-quality "$JPEG_QUALITY"
)

if [[ -n "$LIMIT" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
    ARGS+=(--overwrite)
fi

set +e
python -m dataset_formation.generate "${ARGS[@]}"
EXIT_CODE=$?
set -e

echo "========================================================"
echo "Finished : $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================================"

exit $EXIT_CODE
