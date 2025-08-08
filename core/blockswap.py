"""
Block Swap implementation for ComfyUI-MultiGPU
Based on analysis of WanVideo's block swap mechanism
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gc

@dataclass
class BlockSwapConfig:
    """Configuration for block swapping"""
    virtual_vram_gb: float  # Total model size to offload
    swap_space_gb: float    # Reserved buffer on compute device
    swap_device: str        # Where to offload ("cpu", "cuda:1", etc)
    compute_device: str     # Where to run computation (usually "cuda:0")
    use_non_blocking: bool = False  # Non-blocking transfers
    
    def __post_init__(self):
        self.swap_device = torch.device(self.swap_device)
        self.compute_device = torch.device(self.compute_device)


class BlockSwapManager:
    """Manages block swapping for transformer models"""
    
    def __init__(self, model: torch.nn.Module, config: BlockSwapConfig):
        self.model = model
        self.config = config
        self.blocks = []
        self.current_block_idx = -1
        self.hooks = []
        
        # Calculate model size
        self.model_size_gb = self._calculate_model_size()
        logging.info(f"[BlockSwap] Model size: {self.model_size_gb:.2f} GB")
        
        # Partition model into blocks
        self._partition_model()
        
        # Install hooks
        self._install_hooks()
        
    def _calculate_model_size(self) -> float:
        """Calculate total model size in GB"""
        total_bytes = 0
        for param in self.model.parameters():
            if param.data is not None:
                total_bytes += param.element_size() * param.nelement()
        return total_bytes / (1024**3)
    
    def _get_module_size(self, module: torch.nn.Module) -> float:
        """Calculate size of a module in GB"""
        total_bytes = 0
        for param in module.parameters(recurse=False):
            if param.data is not None:
                total_bytes += param.element_size() * param.nelement()
        return total_bytes / (1024**3)
    
    def _partition_model(self):
        """Partition model into swappable blocks based on swap_space_gb"""
        
        # Find transformer blocks (common patterns)
        transformer = None
        transformer_blocks = []
        
        # Try to find transformer module
        for name, module in self.model.named_modules():
            # Common transformer patterns
            if any(pattern in name.lower() for pattern in ['transformer', 'diffusion_model', 'unet']):
                # Check if it has sequential blocks
                if hasattr(module, 'blocks') or hasattr(module, 'layers'):
                    transformer = module
                    if hasattr(module, 'blocks'):
                        transformer_blocks = list(module.blocks)
                    elif hasattr(module, 'layers'):
                        transformer_blocks = list(module.layers)
                    break
        
        if not transformer_blocks:
            # Fallback: partition all modules
            logging.warning("[BlockSwap] No transformer blocks found, using fallback partitioning")
            self._partition_fallback()
            return
        
        # Group blocks based on swap_space_gb
        current_block = []
        current_size = 0
        swap_space_bytes = self.config.swap_space_gb * (1024**3)
        
        for idx, block in enumerate(transformer_blocks):
            block_size = self._get_module_size(block) * (1024**3)  # Convert to bytes
            
            if current_size + block_size > swap_space_bytes and current_block:
                # Start new block group
                self.blocks.append(current_block)
                current_block = [block]
                current_size = block_size
            else:
                current_block.append(block)
                current_size += block_size
        
        # Add remaining blocks
        if current_block:
            self.blocks.append(current_block)
        
        logging.info(f"[BlockSwap] Partitioned into {len(self.blocks)} block groups")
        for i, group in enumerate(self.blocks):
            group_size = sum(self._get_module_size(b) for b in group)
            logging.info(f"  Block group {i}: {len(group)} blocks, {group_size:.2f} GB")
    
    def _partition_fallback(self):
        """Fallback partitioning when transformer structure is not recognized"""
        all_modules = []
        
        # Collect all modules with parameters
        for name, module in self.model.named_modules():
            if any(param.numel() > 0 for param in module.parameters(recurse=False)):
                all_modules.append((name, module))
        
        # Group by size
        current_block = []
        current_size = 0
        swap_space_bytes = self.config.swap_space_gb * (1024**3)
        
        for name, module in all_modules:
            module_size = self._get_module_size(module) * (1024**3)
            
            if current_size + module_size > swap_space_bytes and current_block:
                self.blocks.append([m for _, m in current_block])
                current_block = [(name, module)]
                current_size = module_size
            else:
                current_block.append((name, module))
                current_size += module_size
        
        if current_block:
            self.blocks.append([m for _, m in current_block])
    
    def _install_hooks(self):
        """Install forward pre-hooks on blocks"""
        for block_idx, block_group in enumerate(self.blocks):
            for module in block_group:
                hook = module.register_forward_pre_hook(
                    lambda m, i, bidx=block_idx: self._pre_forward_hook(m, i, bidx)
                )
                self.hooks.append(hook)
    
    def _pre_forward_hook(self, module: torch.nn.Module, inputs: Tuple, block_idx: int):
        """Hook called before forward pass of each block"""
        if block_idx != self.current_block_idx:
            self._swap_blocks(self.current_block_idx, block_idx)
            self.current_block_idx = block_idx
        return inputs
    
    def _swap_blocks(self, old_idx: int, new_idx: int):
        """Swap blocks between devices"""
        logging.debug(f"[BlockSwap] Swapping from block {old_idx} to {new_idx}")
        
        # Offload old block
        if old_idx >= 0 and old_idx < len(self.blocks):
            for module in self.blocks[old_idx]:
                self._move_module(module, self.config.swap_device)
        
        # Load new block
        if new_idx >= 0 and new_idx < len(self.blocks):
            for module in self.blocks[new_idx]:
                self._move_module(module, self.config.compute_device)
        
        # Clear cache if needed
        if self.config.compute_device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _move_module(self, module: torch.nn.Module, device: torch.device):
        """Move a module to specified device"""
        module.to(device, non_blocking=self.config.use_non_blocking)
    
    def prepare(self):
        """Prepare model for inference by moving all blocks to swap device"""
        logging.info(f"[BlockSwap] Moving all blocks to {self.config.swap_device}")
        for block_group in self.blocks:
            for module in block_group:
                self._move_module(module, self.config.swap_device)
        
        # Reset current block
        self.current_block_idx = -1
        
        # Clear GPU cache
        if self.config.compute_device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def cleanup(self):
        """Remove hooks and cleanup"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logging.info("[BlockSwap] Cleanup complete")


class DisTorch:
    """ComfyUI node for block swap configuration"""
    
    @classmethod
    def INPUT_TYPES(cls):
        from .. import get_device_list
        devices = get_device_list()
        
        return {
            "required": {
                "model": ("MODEL",),
                "virtual_vram_gb": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.1,
                    "max": 64.0,
                    "step": 0.1,
                    "tooltip": "Amount of model to offload to swap device"
                }),
                "swap_space_gb": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 16.0,
                    "step": 0.1,
                    "tooltip": "Size of buffer on compute device for active blocks"
                }),
                "swap_device": (devices, {
                    "default": "cpu",
                    "tooltip": "Device to offload inactive blocks to"
                }),
                "compute_device": (devices, {
                    "default": devices[1] if len(devices) > 1 else devices[0],
                    "tooltip": "Device to run computation on"
                }),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use non-blocking memory transfers (faster but uses more RAM)"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_block_swap"
    CATEGORY = "multigpu"
    
    def apply_block_swap(self, model, virtual_vram_gb: float, swap_space_gb: float, 
                         swap_device: str, compute_device: str, use_non_blocking: bool = False):
        """Apply block swap configuration to model"""
        
        logging.info(f"[DisTorch] Configuring block swap:")
        logging.info(f"  Virtual VRAM: {virtual_vram_gb} GB")
        logging.info(f"  Swap space: {swap_space_gb} GB")
        logging.info(f"  Swap device: {swap_device}")
        logging.info(f"  Compute device: {compute_device}")
        
        # Create config
        config = BlockSwapConfig(
            virtual_vram_gb=virtual_vram_gb,
            swap_space_gb=swap_space_gb,
            swap_device=swap_device,
            compute_device=compute_device,
            use_non_blocking=use_non_blocking
        )
        
        # Get the actual model (handle ModelPatcher)
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model
        
        # Check if model has diffusion_model (common pattern)
        if hasattr(actual_model, 'diffusion_model'):
            target_model = actual_model.diffusion_model
        else:
            target_model = actual_model
        
        # Create block swap manager
        manager = BlockSwapManager(target_model, config)
        
        # Prepare model (move blocks to swap device)
        manager.prepare()
        
        # Store manager on model for later access
        model._block_swap_manager = manager
        
        # Also set the load_device attribute if it exists
        if hasattr(model, 'load_device'):
            model.load_device = config.compute_device
        
        logging.info("[DisTorch] Block swap configuration applied successfully")
        
        return (model,)
