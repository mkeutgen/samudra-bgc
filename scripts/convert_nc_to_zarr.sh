#!/bin/bash
#SBATCH --job-name=nc2zarr
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
conda activate clean_zarr_env  

# Print versions (sanity check)
python - << 'EOF'
import zarr, xarray as xr, numcodecs
print("zarr:", zarr.__version__)
print("xarray:", xr.__version__)
print("numcodecs:", numcodecs.__version__)
EOF

# Input glob (adjust to your actual NetCDF path pattern)
IN_GLOB="/scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/prototypes_dataset/ds_mini.nc"

# Output Zarr path
OUT_ZARR="/scratch/gpfs/GEOCLIM/LRGROUP/maximek/INMOS/prototypes_dataset/ds_mini.zarr"

# Run the conversion
python nc_to_zarr.py "$IN_GLOB" "$OUT_ZARR" 