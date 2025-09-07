#!/usr/bin/env python
"""
Preprocess idealized North Atlantic double-gyre BGC data for Samudra training.

What this does:
- Interpolate staggered C-grid vars (u,v,taux,tauy) onto tracer grid (xh,yh)
- Rename to CF-ish names: u->uo, v->vo, taux->tauuo, tauy->tauvo
- Create impermeable walls via a boundary ring (mask_*, wetmask)
- Split 3D vars into per-level channels: var_0, var_1, ...
- Write Zarr v2 (zarr_format=2) with zstd compression + consolidated metadata
- Compute GLOBAL (0-D) means/stds per variable (Samudra-style)
"""

import os
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Blosc
import zarr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------
# Helpers
# ---------------------------
def open_zarr_smart(path: str) -> xr.Dataset:
    """Open a Zarr store with consolidated metadata if available."""
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def interp_to_tracer(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolate staggered variables to tracer grid.
    - If u has xq -> interpolate along xq to xh
    - If v has yq -> interpolate along yq to yh
    - Likewise for taux/tauy if staggered
    """
    logging.info("Interpolating staggered variables (if present) to tracer grid...")
    out = ds

    # velocities
    if "u" in out:
        da = out["u"]
        if "xq" in da.dims:
            out["u"] = da.interp(xq=out["xh"])
            logging.info("  u: xq -> xh interp done")
    if "v" in out:
        da = out["v"]
        if "yq" in da.dims:
            out["v"] = da.interp(yq=out["yh"])
            logging.info("  v: yq -> yh interp done")

    # wind stress
    if "taux" in out:
        da = out["taux"]
        if "xq" in da.dims:
            out["taux"] = da.interp(xq=out["xh"])
            logging.info("  taux: xq -> xh interp done")
    if "tauy" in out:
        da = out["tauy"]
        if "yq" in da.dims:
            out["tauy"] = da.interp(yq=out["yh"])
            logging.info("  tauy: yq -> yh interp done")

    return out


def rename_cf(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables to Samudra/CF-style names."""
    rename_map = {
        "u": "uo",
        "v": "vo",
        "taux": "tauuo",
        "tauy": "tauvo",
    }
    present = {k: v for k, v in rename_map.items() if k in ds}
    if present:
        logging.info(f"Renaming variables: {present}")
        ds = ds.rename(present)
    return ds


def make_closed_basin_masks(ds: xr.Dataset, boundary_width: int = 1) -> xr.Dataset:
    """
    Create masks with an outer land ring (impermeable walls).
    Produces mask_0..mask_{Nz-1} on (yh,xh) and a combined 3D wetmask.
    """
    assert "yh" in ds.dims and "xh" in ds.dims, "Expected tracer grid (yh,xh) in dims"
    assert "z_l" in ds.dims, "Expected vertical dim 'z_l'"

    Ny, Nx, Nz = ds.sizes["yh"], ds.sizes["xh"], ds.sizes["z_l"]
    logging.info(f"Building closed-basin masks with boundary_width={boundary_width}")

    base = np.ones((Ny, Nx), dtype=np.float32)
    if boundary_width > 0:
        bw = boundary_width
        base[:bw, :] = 0.0  # south
        base[-bw:, :] = 0.0 # north
        base[:, :bw] = 0.0  # west
        base[:, -bw:] = 0.0 # east

    for k in range(Nz):
        ds[f"mask_{k}"] = xr.DataArray(base.copy(), dims=("yh", "xh"))

    wet3d = np.stack([base] * Nz, axis=0)  # [z_l,yh,xh]
    ds["wetmask"] = xr.DataArray(
        wet3d, dims=("z_l", "yh", "xh"),
        coords={"z_l": ds["z_l"]}
    )
    return ds


def split_3d(ds: xr.Dataset, var: str, zdim: str = "z_l") -> xr.Dataset:
    """Split 3D var(time,z,y,x) into per-level channels var_0..var_{Nz-1}."""
    if var not in ds:
        return ds
    Nz = ds.sizes.get(zdim, 0)
    for k in range(Nz):
        ds[f"{var}_{k}"] = ds[var].isel({zdim: k})
    ds = ds.drop_vars(var)
    return ds


def write_zarr_v2(ds: xr.Dataset, store: str, clevel: int = 1):
    """Write dataset to Zarr v2 with zstd compression and consolidate metadata."""
    compressor = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    enc = {v: {"compressor": compressor, "dtype": "float32"} for v in ds.data_vars}
    ds32 = ds.astype({v: "float32" for v in ds.data_vars})

    logging.info(f"Writing Zarr v2 -> {store}")
    # zarr_format (new) instead of deprecated zarr_version
    ds32.to_zarr(store, mode="w", consolidated=False, zarr_format=2, encoding=enc)
    zarr.consolidate_metadata(store)
    logging.info("  consolidated metadata")


def compute_global_stats(ds: xr.Dataset, outdir: Path, clevel: int = 1):
    """
    Compute GLOBAL (0-D) mean/std per variable:
    mean over ALL dims (including time), yielding scalars (Samudra-style).
    """
    logging.info("Computing GLOBAL (0-D) means/stds per variable...")
    all_dims = tuple(ds.dims.keys())
    mean_ds = ds.mean(dim=all_dims, keep_attrs=True)
    std_ds  = ds.std(dim=all_dims,  keep_attrs=True, ddof=0)

    mean_store = str(outdir / "bgc_means.zarr")
    std_store  = str(outdir / "bgc_stds.zarr")
    write_zarr_v2(mean_ds, mean_store, clevel=clevel)
    write_zarr_v2(std_ds,  std_store,  clevel=clevel)

    # sanity
    m = xr.open_zarr(mean_store); s = xr.open_zarr(std_store)
    logging.info(f"means dims: {m.dims}  | stds dims: {s.dims}  (expect both = {{}})")
    return mean_store, std_store


# ---------------------------
# Main pipeline
# ---------------------------
def prepare_idealized_bgc_data(
    input_path: str,
    output_dir: str,
    years_to_use: int = 9,
    boundary_width: int = 1,
    clevel: int = 1,
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Opening input Zarr: {input_path}")
    ds = open_zarr_smart(input_path)

    # Optional: subset years (assumes daily cadence)
    if "time" in ds.dims:
        nsteps = years_to_use * 365
        if ds.sizes["time"] > nsteps:
            logging.info(f"Selecting first {years_to_use} years ({nsteps} steps)")
            ds = ds.isel(time=slice(0, nsteps))

    # Interpolate off staggered grid (if needed)
    ds = interp_to_tracer(ds)

    # Rename to CF-style names
    ds = rename_cf(ds)

    # Optional convenience: surface layer of thkcello
    if "thkcello" in ds and "z_l" in ds.dims:
        ds["thkcello_surface"] = ds["thkcello"].isel(z_l=0)

    # Closed-basin masks (impermeable walls)
    ds = make_closed_basin_masks(ds, boundary_width=boundary_width)

    # Split 3D variables into level channels (only if they exist)
    for base in ["CT", "SA", "o2", "dic", "chl", "pp", "uo", "vo"]:
        ds = split_3d(ds, base, zdim="z_l")

    # Choose variables to keep
    vars_3d = []
    if "z_l" in ds.dims:
        Nz = ds.sizes["z_l"]
        for base in ["CT", "SA", "o2", "dic", "chl", "pp", "uo", "vo"]:
            for k in range(Nz):
                name = f"{base}_{k}"
                if name in ds:
                    vars_3d.append(name)

    extras = [v for v in ["tauuo", "tauvo", "Qnet", "PRCmE", "thkcello_surface"] if v in ds]
    masks  = [f"mask_{k}" for k in range(ds.sizes.get("z_l", 0)) if f"mask_{k}" in ds] + ["wetmask"]
    keep   = [v for v in (vars_3d + extras + masks) if v in ds]
    ds_out = ds[keep]

    # >>> Rechunk to legal, performant chunks (fixes "final chunk > first chunk" error) <<<
    # Choose chunks that keep last chunk <= first chunk on each axis and reduce file count.
    # Adjust as desired; these are safe defaults for ~daily data on ~50x180x270 grids.
    chunk_map = {}
    if "time" in ds_out.dims:
        chunk_map["time"] = 62      # any value >= any trailing remainder; improves I/O for training
    if "z_l" in ds_out.dims:
        chunk_map["z_l"]  = 10
    if "yh" in ds_out.dims:
        chunk_map["yh"]   = 180
    if "xh" in ds_out.dims:
        chunk_map["xh"]   = 180
    if chunk_map:
        ds_out = ds_out.chunk(chunk_map)

    # Write processed data
    data_store = str(outdir / "bgc_data.zarr")
    write_zarr_v2(ds_out, data_store, clevel=clevel)

    # GLOBAL (0-D) stats (exclude masks from stats)
    compute_global_stats(ds_out.drop_vars(masks, errors="ignore"), outdir, clevel=clevel)

    logging.info("Preprocessing complete.")
    return data_store


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Prepare idealized North Atlantic BGC data for Samudra")
    p.add_argument("--input",  required=True, help="Path to ds_mini.zarr")
    p.add_argument("--output", required=True, help="Directory to write processed outputs")
    p.add_argument("--years",  type=int, default=9, help="Years to use (assumes daily cadence)")
    p.add_argument("--bw",     type=int, default=1, help="Boundary ring width (cells) for closed basin")
    p.add_argument("--clevel", type=int, default=1, help="zstd compression level (1=fast)")
    args = p.parse_args()

    prepare_idealized_bgc_data(
        input_path=args.input,
        output_dir=args.output,
        years_to_use=args.years,
        boundary_width=args.bw,
        clevel=args.clevel,
    )