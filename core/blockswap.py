"""
Block Swap implementation for ComfyUI-MultiGPU
Based on analysis of WanVideo's block swap mechanism
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gc
import os
from datetime import datetime
import traceback
import json

# Set up file logging for DisTorch
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"distorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure file handler
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d - [%(name)s] - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Create DisTorch logger
distorch_logger = logging.getLogger("DisTorch")
distorch_logger.setLevel(logging.DEBUG)
distorch_logger.addHandler(file_handler)

# Also add console handler for important messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[DisTorch] %(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
distorch_logger.addHandler(console_handler)

distorch_logger.info(f"DisTorch logging initialized. Log file: {log_file}")
distorch_logger.debug("="*80)
distorch_logger.debug("DISTORCH BLOCK SWAP MODULE LOADED")
distorch_logger.debug(f"PyTorch version: {torch.__version__}")
distorch_logger.debug(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    distorch_logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        distorch_logger.debug(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        distorch_logger.debug(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
distorch_logger.debug("="*80)


@dataclass
class BlockSwapConfig:
    """Configuration for block swapping"""
    virtual_vram_gb: float  # Total model size to offload
    swap_space_gb: float    # Reserved buffer on compute device
    swap_device: str        # Where to offload ("cpu", "cuda:1", etc)
    compute_device: str     # Where to run computation (usually "cuda:0")
    use_non_blocking: bool = False  # Non-blocking transfers
    
    def __post_init__(self):
        distorch_logger.debug(f"BlockSwapConfig.__post_init__ called")
        distorch_logger.debug(f"  Raw swap_device: {self.swap_device}")
        distorch_logger.debug(f"  Raw compute_device: {self.compute_device}")
        distorch_logger.debug(f"  virtual_vram_gb: {self.virtual_vram_gb}")
        distorch_logger.debug(f"  swap_space_gb: {self.swap_space_gb}")
        distorch_logger.debug(f"  use_non_blocking: {self.use_non_blocking}")
        
        try:
            self.swap_device = torch.device(self.swap_device)
            self.compute_device = torch.device(self.compute_device)
            distorch_logger.debug(f"  Converted swap_device: {self.swap_device}")
            distorch_logger.debug(f"  Converted compute_device: {self.compute_device}")
        except Exception as e:
            distorch_logger.error(f"Error converting devices: {e}")
            distorch_logger.error(traceback.format_exc())
            raise


class BlockSwapManager:
    """Manages block swapping for transformer models"""
    
    def __init__(self, model: torch.nn.Module, config: BlockSwapConfig):
        distorch_logger.debug("BlockSwapManager.__init__ called")
        distorch_logger.debug(f"  Model type: {type(model).__name__}")
        distorch_logger.debug(f"  Model device: {next(model.parameters()).device if any(model.parameters()) else 'no params'}")
        
        self.model = model
        self.config = config
        self.blocks = []
        self.current_block_idx = -1
        self.hooks = []
        self.swap_count = 0  # Track number of swaps for debugging
        
        # Calculate model size
        distorch_logger.debug("Calculating model size...")
        self.model_size_gb = self._calculate_model_size()
        distorch_logger.info(f"Model size: {self.model_size_gb:.2f} GB")
        
        # Log model structure
        self._log_model_structure()
        
        # Partition model into blocks
        distorch_logger.debug("Partitioning model into blocks...")
        self._partition_model()
        
        # Install hooks
        distorch_logger.debug("Installing forward hooks...")
        self._install_hooks()
        
        distorch_logger.debug("BlockSwapManager initialization complete")
        
    def _log_model_structure(self):
        """Log the model structure for debugging"""
        distorch_logger.debug("Model structure analysis:")
        module_count = 0
        param_count = 0
        module_types = {}
        
        for name, module in self.model.named_modules():
            module_count += 1
            module_type = type(module).__name__
            module_types[module_type] = module_types.get(module_type, 0) + 1
            
            # Count parameters in this module
            module_params = sum(p.numel() for p in module.parameters(recurse=False))
            if module_params > 0:
                param_count += module_params
                if module_count <= 10:  # Log first 10 modules with params
                    distorch_logger.debug(f"  {name}: {module_type} ({module_params:,} params)")
        
        distorch_logger.debug(f"Total modules: {module_count}")
        distorch_logger.debug(f"Total parameters: {param_count:,}")
        distorch_logger.debug("Module type distribution:")
        for module_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            distorch_logger.debug(f"  {module_type}: {count}")
        
    def _calculate_model_size(self) -> float:
        """Calculate total model size in GB"""
        total_bytes = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.data is not None:
                param_bytes = param.element_size() * param.nelement()
                total_bytes += param_bytes
                param_count += 1
                
                if param_count <= 5:  # Log first 5 parameters
                    distorch_logger.debug(f"  Param {name}: shape={param.shape}, bytes={param_bytes:,}")
        
        distorch_logger.debug(f"Total parameters: {param_count}")
        distorch_logger.debug(f"Total bytes: {total_bytes:,}")
        
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
        distorch_logger.debug("Starting model partitioning...")
        
        # Find transformer blocks (common patterns)
        transformer = None
        transformer_blocks = []
        
        # Try to find transformer module
        for name, module in self.model.named_modules():
            # Common transformer patterns
            patterns = ['transformer', 'diffusion_model', 'unet', 'encoder', 'decoder']
            if any(pattern in name.lower() for pattern in patterns):
                distorch_logger.debug(f"Found potential transformer module: {name} ({type(module).__name__})")
                
                # Check if it has sequential blocks
                if hasattr(module, 'blocks'):
                    transformer = module
                    transformer_blocks = list(module.blocks)
                    distorch_logger.debug(f"  Found {len(transformer_blocks)} blocks in {name}.blocks")
                    break
                elif hasattr(module, 'layers'):
                    transformer = module
                    transformer_blocks = list(module.layers)
                    distorch_logger.debug(f"  Found {len(transformer_blocks)} layers in {name}.layers")
                    break
                elif hasattr(module, 'transformer_blocks'):
                    transformer = module
                    transformer_blocks = list(module.transformer_blocks)
                    distorch_logger.debug(f"  Found {len(transformer_blocks)} transformer_blocks in {name}")
                    break
        
        if not transformer_blocks:
            # Fallback: partition all modules
            distorch_logger.warning("No transformer blocks found, using fallback partitioning")
            self._partition_fallback()
            return
        
        # Group blocks based on swap_space_gb
        current_block = []
        current_size = 0
        swap_space_bytes = self.config.swap_space_gb * (1024**3)
        
        distorch_logger.debug(f"Grouping {len(transformer_blocks)} blocks with swap_space={self.config.swap_space_gb} GB")
        
        for idx, block in enumerate(transformer_blocks):
            block_size = self._get_module_size(block) * (1024**3)  # Convert to bytes
            distorch_logger.debug(f"  Block {idx}: size={block_size/(1024**3):.3f} GB")
            
            if current_size + block_size > swap_space_bytes and current_block:
                # Start new block group
                self.blocks.append(current_block)
                distorch_logger.debug(f"    Created block group {len(self.blocks)-1} with {len(current_block)} blocks, total size={current_size/(1024**3):.3f} GB")
                current_block = [block]
                current_size = block_size
            else:
                current_block.append(block)
                current_size += block_size
        
        # Add remaining blocks
        if current_block:
            self.blocks.append(current_block)
            distorch_logger.debug(f"    Created final block group {len(self.blocks)-1} with {len(current_block)} blocks, total size={current_size/(1024**3):.3f} GB")
        
        distorch_logger.info(f"Partitioned into {len(self.blocks)} block groups")
        for i, group in enumerate(self.blocks):
            group_size = sum(self._get_module_size(b) for b in group)
            distorch_logger.info(f"  Block group {i}: {len(group)} blocks, {group_size:.3f} GB")
    
    def _partition_fallback(self):
        """Fallback partitioning when transformer structure is not recognized"""
        distorch_logger.debug("Using fallback partitioning strategy")
        all_modules = []
        
        # Collect all modules with parameters
        for name, module in self.model.named_modules():
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            if param_count > 0:
                module_size = self._get_module_size(module)
                all_modules.append((name, module, module_size))
                if len(all_modules) <= 10:
                    distorch_logger.debug(f"  Module {name}: {param_count:,} params, {module_size:.3f} GB")
        
        distorch_logger.debug(f"Found {len(all_modules)} modules with parameters")
        
        # Group by size
        current_block = []
        current_size = 0
        swap_space_bytes = self.config.swap_space_gb * (1024**3)
        
        for name, module, module_size_gb in all_modules:
            module_size = module_size_gb * (1024**3)
            
            if current_size + module_size > swap_space_bytes and current_block:
                self.blocks.append([m for _, m, _ in current_block])
                distorch_logger.debug(f"  Created block group with {len(current_block)} modules, size={current_size/(1024**3):.3f} GB")
                current_block = [(name, module, module_size_gb)]
                current_size = module_size
            else:
                current_block.append((name, module, module_size_gb))
                current_size += module_size
        
        if current_block:
            self.blocks.append([m for _, m, _ in current_block])
            distorch_logger.debug(f"  Created final block group with {len(current_block)} modules, size={current_size/(1024**3):.3f} GB")
    
    def _install_hooks(self):
        """Install forward pre-hooks on blocks"""
        distorch_logger.debug(f"Installing hooks on {len(self.blocks)} block groups")
        
        for block_idx, block_group in enumerate(self.blocks):
            distorch_logger.debug(f"  Installing hooks for block group {block_idx} ({len(block_group)} modules)")
            for module in block_group:
                hook = module.register_forward_pre_hook(
                    lambda m, i, bidx=block_idx: self._pre_forward_hook(m, i, bidx)
                )
                self.hooks.append(hook)
        
        distorch_logger.debug(f"Installed {len(self.hooks)} hooks total")
    
    def _pre_forward_hook(self, module: torch.nn.Module, inputs: Tuple, block_idx: int):
        """Hook called before forward pass of each block"""
        if block_idx != self.current_block_idx:
            distorch_logger.debug(f"Pre-forward hook triggered: current={self.current_block_idx}, needed={block_idx}")
            self._swap_blocks(self.current_block_idx, block_idx)
            self.current_block_idx = block_idx
        return inputs
    
    def _swap_blocks(self, old_idx: int, new_idx: int):
        """Swap blocks between devices"""
        self.swap_count += 1
        distorch_logger.debug(f"[Swap #{self.swap_count}] Swapping from block {old_idx} to {new_idx}")
        distorch_logger.debug(f"  Swap device: {self.config.swap_device}")
        distorch_logger.debug(f"  Compute device: {self.config.compute_device}")
        
        start_time = datetime.now()
        
        # Offload old block
        if old_idx >= 0 and old_idx < len(self.blocks):
            distorch_logger.debug(f"  Offloading block {old_idx} to {self.config.swap_device}")
            for i, module in enumerate(self.blocks[old_idx]):
                self._move_module(module, self.config.swap_device)
                if i == 0:  # Log first module movement
                    distorch_logger.debug(f"    Moved module {type(module).__name__} to {self.config.swap_device}")
        
        # Load new block
        if new_idx >= 0 and new_idx < len(self.blocks):
            distorch_logger.debug(f"  Loading block {new_idx} to {self.config.compute_device}")
            for i, module in enumerate(self.blocks[new_idx]):
                self._move_module(module, self.config.compute_device)
                if i == 0:  # Log first module movement
                    distorch_logger.debug(f"    Moved module {type(module).__name__} to {self.config.compute_device}")
        
        # Clear cache if needed
        if self.config.compute_device.type == 'cuda':
            before_free = torch.cuda.memory_reserved(self.config.compute_device) / (1024**3)
            torch.cuda.empty_cache()
            after_free = torch.cuda.memory_reserved(self.config.compute_device) / (1024**3)
            distorch_logger.debug(f"  GPU cache cleared: {before_free:.2f} GB -> {after_free:.2f} GB")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        distorch_logger.debug(f"  Swap completed in {elapsed:.3f} seconds")
    
    def _move_module(self, module: torch.nn.Module, device: torch.device):
        """Move a module to specified device"""
        try:
            module.to(device, non_blocking=self.config.use_non_blocking)
        except Exception as e:
            distorch_logger.error(f"Error moving module {type(module).__name__} to {device}: {e}")
            distorch_logger.error(traceback.format_exc())
            raise
    
    def prepare(self):
        """Prepare model for inference by moving all blocks to swap device"""
        distorch_logger.info(f"Preparing model: moving all blocks to {self.config.swap_device}")
        
        start_time = datetime.now()
        
        for i, block_group in enumerate(self.blocks):
            distorch_logger.debug(f"  Moving block group {i} ({len(block_group)} modules)")
            for module in block_group:
                self._move_module(module, self.config.swap_device)
        
        # Reset current block
        self.current_block_idx = -1
        
        # Clear GPU cache
        if self.config.compute_device.type == 'cuda':
            before_free = torch.cuda.memory_reserved(self.config.compute_device) / (1024**3)
            torch.cuda.empty_cache()
            after_free = torch.cuda.memory_reserved(self.config.compute_device) / (1024**3)
            distorch_logger.debug(f"GPU memory after prepare: {before_free:.2f} GB -> {after_free:.2f} GB")
        
        gc.collect()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        distorch_logger.info(f"Model preparation completed in {elapsed:.3f} seconds")
    
    def cleanup(self):
        """Remove hooks and cleanup"""
        distorch_logger.info("Cleaning up BlockSwapManager")
        distorch_logger.debug(f"  Total swaps performed: {self.swap_count}")
        
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        distorch_logger.info("BlockSwap cleanup complete")


class DisTorch:
    """ComfyUI node for block swap configuration"""
    
    @classmethod
    def INPUT_TYPES(cls):
        from .. import get_device_list
        devices = get_device_list()
        
        distorch_logger.debug(f"DisTorch.INPUT_TYPES called, available devices: {devices}")
        
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
        
        distorch_logger.info("="*60)
        distorch_logger.info("DisTorch.apply_block_swap called")
        distorch_logger.info("="*60)
        
        distorch_logger.info(f"Configuration:")
        distorch_logger.info(f"  Virtual VRAM: {virtual_vram_gb} GB")
        distorch_logger.info(f"  Swap space: {swap_space_gb} GB")
        distorch_logger.info(f"  Swap device: {swap_device}")
        distorch_logger.info(f"  Compute device: {compute_device}")
        distorch_logger.info(f"  Non-blocking: {use_non_blocking}")
        
        distorch_logger.debug(f"Input model type: {type(model)}")
        distorch_logger.debug(f"Model attributes: {dir(model)[:10]}...")  # Log first 10 attributes
        
        try:
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
                distorch_logger.debug(f"Extracted actual model from ModelPatcher: {type(actual_model)}")
            else:
                actual_model = model
                distorch_logger.debug(f"Using model directly: {type(actual_model)}")
            
            # Check if model has diffusion_model (common pattern)
            if hasattr(actual_model, 'diffusion_model'):
                target_model = actual_model.diffusion_model
                distorch_logger.debug(f"Found diffusion_model: {type(target_model)}")
            else:
                target_model = actual_model
                distorch_logger.debug(f"No diffusion_model found, using model as-is")
            
            # Create block swap manager
            distorch_logger.debug("Creating BlockSwapManager...")
            manager = BlockSwapManager(target_model, config)
            
            # Prepare model (move blocks to swap device)
            distorch_logger.debug("Preparing model...")
            manager.prepare()
            
            # Store manager on model for later access
            model._block_swap_manager = manager
            distorch_logger.debug("Stored BlockSwapManager on model._block_swap_manager")
            
            # Also set the load_device attribute if it exists
            if hasattr(model, 'load_device'):
                model.load_device = config.compute_device
                distorch_logger.debug(f"Set model.load_device to {config.compute_device}")
            
            distorch_logger.info("Block swap configuration applied successfully")
            distorch_logger.info("="*60)
            
        except Exception as e:
            distorch_logger.error(f"Error in apply_block_swap: {e}")
            distorch_logger.error(traceback.format_exc())
            raise
        
        return (model,)
