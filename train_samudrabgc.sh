#!/bin/bash
#SBATCH --job-name=train_samudra_bgc
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                    # More CPUs for data loading with 300GB dataset
#SBATCH --mem=512G                           # More memory for large dataset (not per-GPU)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu
#SBATCH --time=24:00:00

# Load modules 
module purge
module load anaconda3/2024.10               
# Activate your environment
conda activate samudra

cd /scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/samudra-bgc  # Add explicit path


# Launch training with correct number of GPUs
torchrun \
  --nproc_per_node=2 \
  src/train.py \
  --config configs/train_bgc_config.yaml
