#!/bin/bash
#SBATCH --job-name=train_samudra_bgc_fast
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=4                           # One task per GPU for distributed training
#SBATCH --cpus-per-task=8                    # 8 CPUs per GPU (32 total)
#SBATCH --mem=512G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu
#SBATCH --time=48:00:00

# Load modules 
module purge
module load anaconda3/2024.10               

# Activate environment
conda activate samudra
cd /scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/samudra-bgc

# CRITICAL PERFORMANCE OPTIMIZATIONS
export OMP_NUM_THREADS=8                    # Match cpus-per-task
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# GPU and distributed training optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN                      # Less verbose than INFO
export NCCL_IB_DISABLE=1                    # Disable InfiniBand if causing issues
export NCCL_P2P_DISABLE=1                   # May help with multi-GPU stability

# Memory and I/O optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Better memory management
export HDF5_USE_FILE_LOCKING=FALSE          # Prevent file locking issues
export TMPDIR=/scratch/gpfs/GEOCLIM/LRGROUP/maximek/tmp  # Fast scratch space

# Create temp directory
mkdir -p $TMPDIR

# Launch with optimized settings
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
  src/train.py \
  --config configs/train_bgc_config_optimized.yaml