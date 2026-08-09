#!/bin/bash

# SLURM job terminal commands->
# Submit  : sbatch jobScheduler_prajna.sh
# Monitor : squeue --me
# Cancel  : scancel <jobid>
# Notes:
# - Adjust resource requests (CPUs, memory, time) based on your job's needs for Prajna cluster.
# - It has three partitions->
# - l40  → L40S GPU,  8 GPUs/node,  7 nodes,  56 GPUs total,  max 2 days
# - a40  → A40 GPU,   4 GPUs/node, 20 nodes,  80 GPUs total,  max 4 days
# - dgx  → A100 GPU,  8 GPUs/node,  9 nodes,  72 GPUs total,  max 6 days
# - Partition & QOS must match — AllowQos confirmed from scontrol show partition


#SBATCH --job-name=my_exp               # Job name (change to your job/experiment name)
#SBATCH --output=logs/my_exp-%j.out     # Standard output log (with job ID in filename); "%j" is replaced by the job ID
#SBATCH --error=logs/my_exp-%j.err      # Error log (with job ID in filename); "%j" is replaced by the job ID
#SBATCH --nodes=1                       # Number of nodes (1 for single-node jobs) -> Generally do not change for single-node jobs
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for Python scripts)
#SBATCH --cpus-per-task=8               # Cores for data loading/preprocessing (max 32 on l40)
#SBATCH --mem=64G                       # Memory per node (adjust based on your needs)
#SBATCH --time=24:00:00                 # Max time: l40=2d | a40=4d | dgx=6d (format: HH:MM:SS)
#SBATCH --partition=l40                 # Partition name (l40, a40, or dgx) -> GPU in needed 
#SBATCH --qos=l40                       # Must match partition: l40 | a40 | dgx
#SBATCH --gres=gpu:1                    # GPUs requested (max per node: l40=8, a40=4, dgx=8)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
# Replace with your actual conda environment name
conda activate myenv                   

# Change to your project directory (adjust path as needed); mkdir for logs if not exists
cd /home/<groupname>/<username>/my_project   
mkdir -p logs

echo "========================================================"
echo "Starting job... >>>"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUS: $SLURM_CPUS_PER_TASK"
echo "========================================================"

# Replace with your actual command to run the training script; redirect output to log files
python3 train.py

echo "========================================================"
echo ">>> Job finished."