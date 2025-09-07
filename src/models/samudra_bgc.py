"""
Modified Samudra model for biogeochemistry emulation.
Includes custom output layer with physical constraints.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from models.blocks import ConvNeXtBlock, BilinearUpsample, TransposedConvUpsample
from models.factory import create_block, create_downsample, create_upsample, get_activation_cl
from datasets import InferenceDataset, TrainData
from utils.device import get_device
from utils.train import pairwise


class SamudraBGC(nn.Module):
    """
    Samudra model adapted for biogeochemistry with physical constraints.
    """
    
    def __init__(
        self, 
        config,
        hist: int,
        wet: torch.Tensor,
        prognostic_vars: Optional[List[str]] = None,
        non_negative_vars: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.ch_width = config.ch_width
        self.N_in = config.ch_width[0]
        self.N_out = config.n_out
        self.wet = wet.bool()
        
        print(f"SamudraBGC init - Wet mask shape: {self.wet.shape}")
        print(f"Expected spatial dims based on data: y=270, x=180")

        
        self.N_pad = int((config.last_kernel_size - 1) / 2)
        self.pad = config.pad
        self.hist = hist
        self.input_channels = config.ch_width[0]
        self.prognostic_channels = config.n_out
        
        # Store variable names for corrector
        self.prognostic_vars = prognostic_vars
        self.non_negative_vars = non_negative_vars or []
        
        # Add non-negative variables from config if specified
        if hasattr(config, 'corrector') and hasattr(config.corrector, 'non_negative_corrector_names'):
            self.non_negative_vars.extend(config.corrector.non_negative_corrector_names)
        
        # Get activation class
        activation = get_activation_cl(config.core_block.activation)
        
        # Create local copies of config lists
        ch_width = config.ch_width.copy()
        dilation = config.dilation.copy()
        n_layers = config.n_layers.copy()
        
        # Build encoder (going down)
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            # Core block
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                )
            )
            # Downsampling
            layers.append(create_downsample(config.down_sampling_block))
        
        # Middle block (bottleneck)
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
            )
        )
        
        # First upsampling
        layers.append(
            create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
        )
        
        # Reverse for decoder path
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        
        # Build decoder (going up)
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                create_block(
                    config.core_block.block_type,
                    in_channels=a,
                    out_channels=b,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=config.pad,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                )
            )
            layers.append(
                create_upsample(config.up_sampling_block, in_channels=b, out_channels=b)
            )
        
        # Final conv block before output
        layers.append(
            create_block(
                config.core_block.block_type,
                in_channels=b,
                out_channels=b,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=config.pad,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
            )
        )
        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(config.ch_width) - 1)
        
        # Custom output layer with biogeochemistry corrections
        self.output_layer = self._create_output_layer(b, config.n_out, config.last_kernel_size)
        
    def _create_output_layer(self, in_channels: int, out_channels: int, kernel_size: int):
        """Create output layer with optional corrections for biogeochemistry."""
        
        # If we have non-negative variables, create indices for them
        if self.prognostic_vars and self.non_negative_vars:
            non_neg_indices = []
            for i, var in enumerate(self.prognostic_vars):
                var_name = var.split('_')[0]
                if var_name in self.non_negative_vars:
                    non_neg_indices.append(i)
            self.register_buffer('non_neg_indices', torch.tensor(non_neg_indices, dtype=torch.long))
        else:
            self.non_neg_indices = None
        
        # Standard convolution for output
        padding = kernel_size // 2
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    
    def apply_corrections(self, x: torch.Tensor) -> torch.Tensor:
        """Apply physical corrections to output."""
        
        # Apply non-negativity constraints
        if self.non_neg_indices is not None and len(self.non_neg_indices) > 0:
            # Use ReLU to ensure non-negative values
            x[:, self.non_neg_indices] = torch.relu(x[:, self.non_neg_indices])


        # Apply wet mask
        x = torch.where(self.wet, x, 0.0)
        
        return x
    
    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        """Single forward pass through the network."""
        
        # Store skip connections for U-Net architecture
        skip_connections = []
        
        # Encoder path
        for i in range(self.num_steps):
            # Core block
            layer_idx = i * 2
            fts = self.layers[layer_idx](fts)
            skip_connections.append(fts)
            
            # Downsample
            fts = self.layers[layer_idx + 1](fts)
        
        # Bottleneck
        bottleneck_idx = self.num_steps * 2
        fts = self.layers[bottleneck_idx](fts)
        
        # First upsample
        fts = self.layers[bottleneck_idx + 1](fts)
        
        # Decoder path
        decoder_start = bottleneck_idx + 2
        for i in range(self.num_steps):
            # Add skip connection
            skip = skip_connections[self.num_steps - 1 - i]
            
            # Resize and add skip connection (similar to original Samudra)
            if fts.shape != skip.shape:
                crop = np.array(fts.shape[2:])
                shape = np.array(skip.shape[2:])
                pads = shape - crop
                pads = [pads[1]//2, pads[1] - pads[1]//2, pads[0]//2, pads[0] - pads[0]//2]
                fts = nn.functional.pad(fts, pads)
            
            fts += skip
            
            # Apply decoder blocks
            if i < self.num_steps - 1:
                fts = self.layers[decoder_start + i * 2](fts)
                fts = self.layers[decoder_start + i * 2 + 1](fts)
            else:
                fts = self.layers[decoder_start + i * 2](fts)

        # Store original spatial size
        target_height, target_width = 270, 180

            # Final output layer (REMOVE the manual padding)
        fts = self.output_layer(fts)  # Remove the manual padding before this
    
        # Resize to exact target dimensions if needed
        if fts.shape[-2:] != (target_height, target_width):
            fts = torch.nn.functional.interpolate(
            fts, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
    
        # Apply corrections
        fts = self.apply_corrections(fts)

    
        return fts
    
    def forward(
        self,
        train_data: TrainData,
        loss_fn=None,
    ):
        """Multi-step forward pass for training."""
        outputs = []
        loss = torch.tensor(torch.nan)
        
        for step in range(len(train_data)):
            if step == 0:
                input_tensor = train_data.get_initial_input()
            else:
                input_tensor = train_data.merge_prognostic_and_boundary(
                    prognostic=outputs[-1], step=step
                )
            
            pred = self.forward_once(input_tensor)
            
            if loss_fn is not None:
                if torch.isnan(loss).all():
                    loss = loss_fn(pred, train_data.get_label(step))
                else:
                    loss += loss_fn(pred, train_data.get_label(step))
            
            outputs.append(pred)
        
        return loss if loss_fn else outputs
    
    def inference(
        self,
        dataset: InferenceDataset,
        initial_prognostic=None,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ):
        """Inference/rollout mode."""
        outputs = []
        
        for step in range(num_steps):
            logging.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            
            if step == 0 and steps_completed == 0:
                input_tensor = dataset.get_initial_input().to(device=get_device())
            elif step == 0 and steps_completed > 0:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=initial_prognostic,
                    step=steps_completed,
                )
            else:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=outputs[-1],
                    step=steps_completed + step,
                )
            
            pred = self.forward_once(input_tensor)
            outputs.append(pred)
        
        return outputs