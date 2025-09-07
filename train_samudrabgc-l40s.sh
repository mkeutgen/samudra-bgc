#!/bin/bash
#SBATCH --job-name=train_samudra_l40s
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3

# One L40S per node; increase nodes to scale GPUs
#SBATCH --nodes=2                   # <-- set 1 for single-GPU; N for N GPUs (1 per node)
#SBATCH --ntasks-per-node=1         # one rank per node (1 GPU per node)
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32          # data loading; adjust if you see I/O thrash
#SBATCH --mem=128G                  # per node; bump if preprocessing is heavy
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu
# Optional: avoid noisy neighbors
# SBATCH --exclusive

# --- Env ---
module purge
module load anaconda3/2024.10
conda activate samudra

cd /scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/samudra-bgc || exit 1

# --- Runtime/perf knobs (safe defaults for multi-node L40S) ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# PyTorch CUDA allocator: reduce fragmentation on long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9

# Allow fast paths; tune in code too (torch.backends.* allow_tf32/benchmark)
export CUDNN_BENCHMARK=1
export NVIDIA_TF32_OVERRIDE=0

# NCCL across nodes: prefer IB if present, fall back to TCP automatically
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# If you hit IB issues, uncomment to force TCP:
# export NCCL_IB_DISABLE=1

# Dataloader hints (match in your code if you read these)
export DATALOADER_WORKERS=16
export DATALOADER_PREFETCH=4
export DATALOADER_PERSIST=1
export DATALOADER_PIN=1

# --- Rendezvous (master = first node of allocation) ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=${SLURM_NNODES}

# --- Launch (DDP: 1 proc per node, 1 GPU per node) ---
srun --mpi=pmix \
  python -m torch.distributed.run \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/train.py \
    --config configs/train_bgc_config.yaml
