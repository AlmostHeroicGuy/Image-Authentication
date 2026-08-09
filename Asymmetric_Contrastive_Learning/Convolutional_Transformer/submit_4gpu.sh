#!/bin/bash

# ============================================================================
#  SLURM Job Script - Hybrid Forensic Encoder (4x GPU)
# ============================================================================

#SBATCH --job-name=hybrid_forensic_4gpu

#SBATCH --output=logs/forensic-%j.out
#SBATCH --error=logs/forensic-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=dgx
#SBATCH --qos=dgx

#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

#SBATCH --time=24:00:00

set -euo pipefail

# ============================================================================
# Performance Settings
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
# Environment Activation
# ============================================================================

activate_env() {
    local env_name="$1"

    if command -v mamba >/dev/null 2>&1; then
        eval "$(mamba shell hook --shell bash)"
        mamba activate "$env_name"
        return
    fi

    echo "Mamba not found." >&2
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
echo " Hybrid Forensic Encoder - 4 GPU Training"
echo "========================================================"

echo "JOB ID         : ${SLURM_JOB_ID:-N/A}"
echo "NODE           : $(hostname)"
echo "START TIME     : $(date)"

echo "PROJECT DIR    : $PROJECT_DIR"
echo "DATASET        : $DATASET_PATH"

echo "GPUs           : 4"
echo "BATCH/GPU      : $BATCH_PER_GPU"
echo "Effective Batch: $((BATCH_PER_GPU * 4))"

echo "Workers/GPU    : $NUM_WORKERS"

echo "Epochs         : $EPOCHS"

echo "========================================================"

nvidia-smi

# ============================================================================
# Launch Training
# ============================================================================

set +e

torchrun \
    --standalone \
    --nproc_per_node=4 \
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