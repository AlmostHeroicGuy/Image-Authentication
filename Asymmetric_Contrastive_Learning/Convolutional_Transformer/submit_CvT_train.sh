#!/bin/bash

# ============================================================================
#  SLURM Job Script - Hybrid Forensic Encoder (8x A100 DGX)
#  Prajna AI/ML HPC - IIT Bombay
# ============================================================================

#SBATCH --job-name=hybrid_forensic_8gpu
#SBATCH --output=logs/forensic-%j.out
#SBATCH --error=logs/forensic-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=dgx
#SBATCH --qos=dgx

#SBATCH --gres=gpu:8

#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

#SBATCH --time=48:00:00

set -euo pipefail

# ============================================================================
# Performance / NCCL
# ============================================================================

export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export PYTHONUNBUFFERED=1

# ============================================================================
# User Config
# ============================================================================

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ml}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

DATASET_PATH="${DATASET_PATH:-Faces}"

BATCH_PER_GPU="${BATCH_PER_GPU:-48}"

NUM_WORKERS="${NUM_WORKERS:-8}"

LATEST_CKPT="${LATEST_CKPT:-latest_checkpoint.pth}"

EPOCHS="${EPOCHS:-500}"

# ============================================================================
# Conda / Mamba activation
# ============================================================================

activate_env() {
    local env_name="$1"

    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate "$env_name"
        return
    fi

    if command -v mamba >/dev/null 2>&1; then
        eval "$(mamba shell hook --shell bash)"
        mamba activate "$env_name"
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$env_name"
        return
    fi

    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate "$env_name"
        return
    fi

    echo "Could not initialize conda/mamba environment." >&2
    exit 1
}

activate_env "$CONDA_ENV_NAME"

# ============================================================================
# Setup
# ============================================================================

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p checkpoints

echo "========================================================"
echo " Hybrid Forensic Encoder - DGX Training"
echo "========================================================"

echo "JOB ID         : ${SLURM_JOB_ID:-N/A}"
echo "NODE           : $(hostname)"
echo "START TIME     : $(date)"

echo "PROJECT DIR    : $PROJECT_DIR"
echo "DATASET        : $DATASET_PATH"

echo "GPUs           : 8"
echo "BATCH/GPU      : $BATCH_PER_GPU"
echo "Effective Batch: $((BATCH_PER_GPU * 8))"

echo "Workers/GPU    : $NUM_WORKERS"

echo "Epochs         : $EPOCHS"

echo "Checkpoint     : $LATEST_CKPT"

echo "========================================================"

nvidia-smi

# ============================================================================
# Launch
# ============================================================================

set +e

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py \
    --dataset-path "$DATASET_PATH" \
    --batch-per-gpu "$BATCH_PER_GPU" \
    --epochs "$EPOCHS" \
    --warmup-epochs 10 \
    --num-workers "$NUM_WORKERS" \
    --checkpoint-interval 10 \
    --latest-checkpoint "$LATEST_CKPT" \
    --enable-checkpointing \
    "$@"

EXIT_CODE=$?

set -e

# ============================================================================
# Finish
# ============================================================================

echo "========================================================"
echo "Training finished"
echo "END TIME  : $(date)"
echo "EXIT CODE : $EXIT_CODE"
echo "========================================================"

exit $EXIT_CODE