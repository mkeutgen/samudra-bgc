"""
Specialized utilities for idealized North Atlantic double gyre configuration.
"""

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from typing import Optional, Tuple


class DoubleGyreForcing:
    """
    Generate and handle forcing for idealized double gyre configuration.
    """
    
    def __init__(
        self,
        nx: int = 180,  # Grid points in x
        ny: int = 270,  # Grid points in y
        lx: float = 2000.0,  # Domain size in x (km)
        ly: float = 3000.0,  # Domain size in y (km)
        tau0: float = 0.1,  # Maximum wind stress (N/m^2)
        seasonal: bool = True,
        epsilon: float = 0.1  # Asymmetry parameter
    ):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.tau0 = tau0
        self.seasonal = seasonal
        self.epsilon = epsilon
        
        # Create coordinate arrays
        self.x = np.linspace(0, lx, nx)
        self.y = np.linspace(0, ly, ny)
        
    def wind_stress(self, t: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute wind stress for double gyre at