import sys, os
import xarray as xr
import zarr
from numcodecs import Blosc

inp, outp = sys.argv[1], sys.argv[2]
os.makedirs(outp, exist_ok=True)

# Read one or many NetCDFs with dask (wrap single path in a list)
ds = xr.open_mfdataset([inp], combine="by_coords", parallel=True,
                       chunks={"time": 1}, engine="netcdf4")   # CHANGED


# Store as float32 with compression
compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
encoding = {v: {"compressor": compressor, "dtype": "float32"} for v in ds.data_vars}

# Ensure dtypes are float32 (keeps consistency with training)
ds = ds.astype({v: "float32" for v in ds.data_vars})

# after ds = ds.astype(...)
ds = ds.chunk({"time": 32, "z_l": 44, "yh": 270, "xh": 180, "xq": 121, "yq": 121})  # ADDED

# Write consolidated Zarr
ds.to_zarr(outp, mode="w", consolidated=True, encoding=encoding)
zarr.consolidate_metadata(outp)
print("Wrote:", outp)
