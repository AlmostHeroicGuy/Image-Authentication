#!/bin/bash

#==============================================================================
# Realistic Asymmetric Contrastive Learning (4×L40S)
#==============================================================================

#SBATCH --job-name=asym-contrastive

#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

#SBATCH --partition=l40
#SBATCH --qos=l40

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

#SBATCH --time=12:00:00

set -euo pipefail

#==============================================================================
# Environment
#==============================================================================

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=2

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

#==============================================================================
# Activate Environment
#==============================================================================

eval "$(mamba shell hook --shell bash)"
mamba activate ml

#==============================================================================
# Directories
#==============================================================================

PROJECT_DIR="$HOME/Realistic_AsymmetricPrototype"

DATASET_PATH="$HOME/imagenet-100/train"

cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p checkpoints

#==============================================================================
# Training Parameters
#==============================================================================

BATCH_PER_GPU=256
NUM_WORKERS=8
EPOCHS=200

LATEST_CKPT="checkpoints/latest_checkpoint.pth"

#==============================================================================
# Information
#==============================================================================

echo "========================================================"

echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $(hostname)"
echo "Date            : $(date)"

echo "GPUs            : 4"

echo "Batch/GPU       : $BATCH_PER_GPU"
echo "Global Batch    : $((BATCH_PER_GPU * 4))"

echo "Workers/GPU     : $NUM_WORKERS"

echo "Dataset         : $DATASET_PATH"

echo "Epochs          : $EPOCHS"

echo "========================================================"

nvidia-smi

#==============================================================================
# Launch
#==============================================================================

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
    --latest-checkpoint "$LATEST_CKPT"

EXIT_CODE=$?

echo "========================================================"

echo "Finished : $(date)"
echo "Exit Code: $EXIT_CODE"

echo "========================================================"

exit $EXIT_CODE