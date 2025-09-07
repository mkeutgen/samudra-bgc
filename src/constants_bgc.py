import logging
from typing import Dict, Optional

import torch
import xarray as xr

# Prognostic variables: 4 vars × 44 levels (no surface variable)
# - CT: 44 levels (CT_0, CT_1, ..., CT_43)
# - SA: 44 levels (SA_0, SA_1, ..., SA_43) 
# - o2: 44 levels (o2_0, o2_1, ..., o2_43)
# - dic: 44 levels (dic_0, dic_1, ..., dic_43)
# Total prognostic: 4 × 44 = 176

# Boundary variables: 3 (tauuo, tauvo, Qnet)

# Input channels: (hist+1) × prognostic + boundary = (1+1) × 176 + 3 = 355
# Output channels: (hist+1) × prognostic = (1+1) × 176 = 352


# For idealized North Atlantic double gyre configuration
# 44 depth levels from  ds_mini dataset
DEPTH_LEVELS = [  1.   ,   3.   ,   5.   ,   7.   ,   9.   ,  11.   ,  13.   ,  15.005,
        17.015,  19.03 ,  21.055,  23.095,  25.16 ,  27.255,  29.385,  31.565,
        33.81 ,  36.135,  38.56 ,  41.105,  43.795,  46.655,  49.715,  53.015,
        56.6  ,  60.515,  64.805,  69.525,  74.74 ,  80.515,  86.92 ,  94.04 ,
       101.96 , 110.77 , 120.575, 131.485, 143.615, 157.095, 172.06 , 188.655,
       207.035, 227.365, 249.82 , 274.585]

# Generate depth index levels for 44 levels
DEPTH_I_LEVELS = [str(i) for i in range(44)]

# Mask variables for 44 levels
MASK_VARS = [f"mask_{i}" for i in range(44)]

# Prognostic variables for idealized biogeochemistry experiment
PROG_VARS_MAP = {
    # Physical + biogeochemical tracers
    "phys_bgc": [
        k + str(j) for k in ["CT_", "SA_", "o2_", "dic_"] 
        for j in DEPTH_I_LEVELS
    ],
    
    # All available tracers including chlorophyll and primary production
    "full_bgc": [
        k + str(j) for k in ["CT_", "SA_", "o2_", "dic_", "chl_", "pp_"] 
        for j in DEPTH_I_LEVELS
    ],
    
    # Only biogeochemical tracers
    "bgc_only": [
        k + str(j) for k in ["o2_", "dic_", "chl_", "pp_"] 
        for j in DEPTH_I_LEVELS
    ],
    
    # Temperature and biogeochemistry (minimal physics)
    "temp_bgc": [
        k + str(j) for k in ["CT_", "o2_", "dic_"] 
        for j in DEPTH_I_LEVELS
    ],
    
    # With velocities (if you want to include dynamics)
    "dynamic_bgc": [
        k + str(j) for k in ["CT_", "SA_", "uo_", "vo_", "o2_", "dic_"] 
        for j in DEPTH_I_LEVELS
    ],
}

# Boundary variables - surface forcing for double gyre
BOUND_VARS_MAP = {
    "surface_forcing": ["tauuo", "tauvo", "Qnet", "PRCmE"],
    "wind_heat": ["tauuo", "tauvo", "Qnet"],  # Wind-driven double gyre
    "heat_freshwater": ["Qnet", "PRCmE"],
    "full_forcing": ["tauuo", "tauvo", "Qnet", "PRCmE", "thkcello_surface"],
}

default_metadata = {
    "CT": {
        "long_name": "Conservative Temperature",
        "units": r"\degree C",
    },
    "SA": {
        "long_name": "Absolute Salinity",
        "units": "g/kg",
    },
    "uo": {
        "long_name": "Zonal Velocity",
        "units": "m/s",
    },
    "vo": {
        "long_name": "Meridional Velocity",
        "units": "m/s",
    },
    "o2": {
        "long_name": "Dissolved Oxygen",
        "units": "mol/kg",
    },
    "dic": {
        "long_name": "Dissolved Inorganic Carbon",
        "units": "mol/kg",
    },
    "chl": {
        "long_name": "Chlorophyll Concentration",
        "units": "ug/kg",
    },
    "pp": {
        "long_name": "Primary Production",
        "units": "mol m-3 s-1",
    },
    "tauuo": {
        "long_name": "Zonal Wind Stress",
        "units": "N/m^2",
    },
    "tauvo": {
        "long_name": "Meridional Wind Stress",
        "units": "N/m^2",
    },
    "Qnet": {
        "long_name": "Net Surface Heat Flux",
        "units": "W/m^2",
    },
    "PRCmE": {
        "long_name": "Precipitation minus Evaporation",
        "units": "kg m-2 s-1",
    },
    "thkcello": {
        "long_name": "Cell Thickness",
        "units": "m",
    },
}


def construct_metadata(data: xr.Dataset) -> Dict[str, Dict[str, str]]:
    """Construct metadata dictionary from dataset."""
    metadata = {}
    for var in data.variables:
        try:
            metadata[var] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in default_metadata:
                metadata[var] = default_metadata[var]
            elif var.split("_")[0] in default_metadata:
                metadata[var] = default_metadata[var.split("_")[0]]
            else:
                logging.debug(f"{var} does not have any default metadata")
                metadata[var] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }
    return metadata


class TensorMap:
    """Maps input variables/depth levels to tensor indices for idealized config."""
    _instance: Optional["TensorMap"] = None

    def __new__(cls, *args, **kwargs) -> "TensorMap":
        raise TypeError(
            "TensorMap cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "TensorMap":
        if cls._instance is None:
            raise ValueError("TensorMap not initialized")
        return cls._instance

    @classmethod
    def init_instance(
        cls, prognostic_vars_key: str, boundary_vars_key: str
    ) -> "TensorMap":
        if cls._instance is not None:
            raise ValueError("TensorMap already initialized")

        instance = super().__new__(cls)
        instance._initialize(prognostic_vars_key, boundary_vars_key)
        cls._instance = instance
        return cls._instance

    def _initialize(self, prognostic_vars_key: str, boundary_vars_key: str):
        """Initialize tensor mapping for biogeochemistry variables."""
        self.prognostic_vars = PROG_VARS_MAP[prognostic_vars_key]
        self.boundary_vars = BOUND_VARS_MAP[boundary_vars_key]

        # Special handling for velocities on staggered grid
        self.outputs = self.prognostic_vars  # Alias for compatibility
        
        self.VAR_3D_IDX: Dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: Dict[str, torch.Tensor] = {}

        self.VAR_SET_2D = []
        self.VAR_SET_3D = []
        
        for out in self.prognostic_vars:
            var_split = out.split("_")
            if len(var_split) == 1:
                self.VAR_SET_2D.append(var_split[0])
            else:
                self.VAR_SET_3D.append(var_split[0])

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys(([out.split("_")[0] for out in self.prognostic_vars]))
        )
        self.DEPTH_SET = DEPTH_I_LEVELS

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()

    def _populate_var_3d_idx(self):
        for kt in self.VAR_SET:
            self.VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.prognostic_vars):
                if kt in k:
                    self.VAR_3D_IDX[kt] = torch.cat(
                        [self.VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.VAR_3D_IDX[kt] = self.VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        for d in self.DEPTH_SET:
            self.DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.prognostic_vars):
                k_split = k.split("_")
                if len(k_split) == 1:
                    continue
                elif d == k_split[-1]:
                    self.DP_3D_IDX[d] = torch.cat(
                        [self.DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.DP_3D_IDX[d] = self.DP_3D_IDX[d].to(torch.int32)

        if self.VAR_SET_2D:
            self.DP_3D_IDX[self.DEPTH_SET[0]] = torch.cat(
                [
                    self.DP_3D_IDX[self.DEPTH_SET[0]],
                    torch.tensor([self.VAR_3D_IDX[var_2D] for var_2D in self.VAR_SET_2D]),
                ]
            )