"""
Corrector module for biogeochemical variables.
Ensures physical constraints are maintained.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class BiogeochemCorrector(nn.Module):
    """
    Applies physical constraints to biogeochemical model outputs.
    """
    
    def __init__(
        self, 
        prognostic_vars: List[str],
        non_negative_vars: Optional[List[str]] = None,
        wet_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            prognostic_vars: List of all prognostic variable names
            non_negative_vars: Variables that should be non-negative
            wet_mask: Mask for ocean points
        """
        super().__init__()
        
        self.prognostic_vars = prognostic_vars
        self.wet_mask = wet_mask
        
        # Default non-negative variables for biogeochemistry
        if non_negative_vars is None:
            non_negative_vars = ['o2', 'dic', 'chl', 'pp']
        
        # Create indices for non-negative variables
        self.non_neg_indices = []
        for i, var in enumerate(prognostic_vars):
            var_name = var.split('_')[0]
            if var_name in non_negative_vars:
                self.non_neg_indices.append(i)
        
        self.non_neg_indices = torch.tensor(self.non_neg_indices, dtype=torch.long)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply corrections to model output.
        
        Args:
            x: Model output tensor [batch, channels, height, width]
        
        Returns:
            Corrected tensor
        """
        # Apply non-negativity constraint
        if len(self.non_neg_indices) > 0:
            # Use ReLU for non-negative variables
            x[:, self.non_neg_indices] = torch.relu(x[:, self.non_neg_indices])
        
        # Apply wet mask if provided
        if self.wet_mask is not None:
            x = torch.where(self.wet_mask, x, 0.0)
        
        return x


class ConservationCorrector(nn.Module):
    """
    Ensures conservation properties for biogeochemical tracers.
    """
    
    def __init__(
        self,
        prognostic_vars: List[str],
        conserved_groups: Optional[dict] = None
    ):
        """
        Args:
            prognostic_vars: List of prognostic variables
            conserved_groups: Dict mapping group names to variable lists
                             that should sum to a constant
        """
        super().__init__()
        
        self.prognostic_vars = prognostic_vars
        
        # Example: nitrogen conservation between different forms
        if conserved_groups is None:
            conserved_groups = {}
        
        self.conserved_groups = conserved_groups
        
    def forward(self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply conservation corrections.
        
        Args:
            x: Current timestep output
            x_prev: Previous timestep (for computing conservation)
        
        Returns:
            Corrected tensor
        """
        if x_prev is None or not self.conserved_groups:
            return x
        
        # Apply conservation constraints for each group
        for group_name, var_list in self.conserved_groups.items():
            # Get indices for this conservation group
            indices = []
            for i, var in enumerate(self.prognostic_vars):
                if var.split('_')[0] in var_list:
                    indices.append(i)
            
            if indices:
                indices = torch.tensor(indices, device=x.device)
                
                # Compute total from previous timestep
                total_prev = x_prev[:, indices].sum(dim=1, keepdim=True)
                
                # Compute current total
                total_curr = x[:, indices].sum(dim=1, keepdim=True)
                
                # Apply correction factor to maintain conservation
                correction = total_prev / (total_curr + 1e-10)
                x[:, indices] *= correction
        
        return x


class BiogeochemOutputLayer(nn.Module):
    """
    Custom output layer for biogeochemistry that includes corrections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        prognostic_vars: List[str] = None,
        non_negative_vars: List[str] = None,
        wet_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        # Final convolution
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=padding
        )
        
        # Corrector
        self.corrector = BiogeochemCorrector(
            prognostic_vars=prognostic_vars,
            non_negative_vars=non_negative_vars,
            wet_mask=wet_mask
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and corrections."""
        x = self.conv(x)
        x = self.corrector(x)
        return x