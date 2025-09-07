#!/bin/bash
#SBATCH --job-name=preprocess-bgc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=cimes3
#SBATCH --time=48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk0964@princeton.edu

# Load/activate environment
module load anaconda3/2024.10
conda activate samudra-bgc  


python preprocess_bgc.py \
  --input /scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/prototypes_dataset/ds_mini.zarr \
  --output /scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/prototypes_dataset/processed_data \
  --years 9
