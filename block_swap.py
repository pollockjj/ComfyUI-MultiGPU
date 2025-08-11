"""
Block Swap Module for SafeTensor Models
Contains all SafeTensor DisTorch code for block-swap memory optimization
"""

import torch
import logging
import copy
from collections import defaultdict
import comfy.model_management as mm
import torch.nn as nn


def log_memory_usage(device, stage=""):
    """Logs the memory usage of a given device."""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    if device.type == 'cuda':
        stats = torch.cuda.memory_stats(device)
        total_mem = mm.get_total_memory(device)
        allocated = stats['allocated_bytes.all.current']
        reserved = stats['reserved_bytes.all.current']
        
        logging.info(
            f"[MemLog] {stage} - {device}: "
            f"Allocated: {allocated / 1024**2:.2f}MB, "
            f"Reserved: {reserved / 1024**2:.2f}MB, "
            f"Total: {total_mem / 1024**3:.2f}GB"
        )
    elif device.type == 'cpu':
        # Basic CPU memory logging (less detailed than CUDA)
        # This requires psutil, which might not be a dependency.
        # For now, we'll just log that it's a CPU.
        logging.info(f"[MemLog] {stage} - {device}: CPU memory logging is not as detailed.")


class BlockSwapManager:
    """
    Manages block-swapping memory optimization using PyTorch hooks.
    """
    def __init__(self, swap_device='cpu'):
        self.swap_device = torch.device(swap_device)
        # Determine the execution device (e.g., GPU)
        self.active_device = mm.get_torch_device()
        self.active_block = None
        # Use a set of IDs for fast lookup of managed blocks
        self.managed_block_ids = set()
        self.hooks = []

    def move_block(self, block, device):
        """Moves a block to the specified device."""
        # Avoid moving if the target device is 'meta'
        if torch.device(device).type != 'meta':
            try:
                block.to(device)
            except Exception as e:
                print(f"[BlockSwap] Warning: Failed to move block {type(block).__name__} to {device}: {e}")

    def _get_block_device(self, block):
        """Robustly determines the current device of a block."""
        try:
            # Check the device of the first parameter found in the block
            param = next(block.parameters(), None)
            if param is not None:
                return param.device
        except Exception:
            pass
        return None

    # CRITICAL FIX: The hook signature must accept (module, args).
    def before_block_execution(self, block, args):
        """
        Hook function called before a block's execution.
        Implements Sequential Swapping (WanVideoWrapper style).
        """
        block_id = id(block)
        if block_id not in self.managed_block_ids:
            return

        # 1. Handle Offloading (if the active block is changing)
        # CRITICAL FIX: This logic must execute regardless of the current block's device.
        if self.active_block != block:
            # Offload the previous block if it exists and is managed by us
            if self.active_block is not None and id(self.active_block) in self.managed_block_ids:
                # print(f"[BlockSwap] Offloading previous block to {self.swap_device}")
                self.move_block(self.active_block, self.swap_device)
            
            # 2. Handle Loading (only if needed)
            current_device = self._get_block_device(block)

            if current_device is None:
                # Block has no parameters, skip loading
                pass
            elif current_device != self.active_device:
                # print(f"[BlockSwap] Loading current block to {self.active_device}")
                self.move_block(block, self.active_device)
            
            # 3. Update the tracker
            self.active_block = block

    def apply_swap_optimization(self, swappable_blocks):
        """
        Applies the block-swapping optimization using PyTorch forward hooks.
        """
        if not swappable_blocks:
            return

        # print(f"[BlockSwap] Applying optimization to {len(swappable_blocks)} blocks.")

        for block in swappable_blocks:
            if not isinstance(block, nn.Module) or id(block) in self.managed_block_ids:
                continue
                
            # Clean up potential previous manual patches
            if hasattr(block, 'original_forward'):
                try:
                    block.forward = block.original_forward
                    del block.original_forward
                except Exception:
                    pass

            block_id = id(block)
            self.managed_block_ids.add(block_id)

            # Use register_forward_pre_hook for robustness.
            try:
                # CRITICAL FIX: Register the method directly, now that its signature is correct.
                hook = block.register_forward_pre_hook(
                    self.before_block_execution
                )
                self.hooks.append(hook)
            except Exception as e:
                 print(f"[BlockSwap] Warning: Failed to register hook for block {type(block).__name__}: {e}")
                 self.managed_block_ids.remove(block_id)
                 continue

            # Move to CPU initially (if it has parameters)
            if self._get_block_device(block) is not None:
                self.move_block(block, self.swap_device)

    def cleanup(self):
        """Removes hooks and restores the model state."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # We rely on the surrounding environment (ComfyUI) to manage the overall 
        # model placement after sampling, but we ensure the last active block is returned to the GPU if needed.
        if self.active_block is not None and self._get_block_device(self.active_block) != self.active_device:
             self.move_block(self.active_block, self.active_device)

        self.managed_block_ids = set()
        self.active_block = None


def analyze_safetensor_distorch(model, compute_device, swap_device, virtual_vram_gb, all_blocks):
    """Provides a detailed analysis of the block swap configuration, mimicking the GGUF DisTorch style."""
    
    eq_line = "=" * 60
    dash_line = "-" * 60
    
    logging.info(eq_line)
    logging.info("          DisTorch SafeTensor Memory Analysis")
    logging.info(eq_line)

    # Device Allocation Table
    fmt_assign = "{:<12}{:>15}{:>15}{:>15}"
    logging.info(fmt_assign.format("Device", "Role", "Total Mem (GB)", "Config (GB)"))
    logging.info(dash_line)
    
    compute_total_gb = mm.get_total_memory(torch.device(compute_device)) / (1024**3)
    swap_total_gb = mm.get_total_memory(torch.device(swap_device)) / (1024**3)
    
    logging.info(fmt_assign.format(compute_device, "Compute", f"{compute_total_gb:.2f}", ""))
    logging.info(fmt_assign.format(swap_device, "Swap", f"{swap_total_gb:.2f}", f"Offload: {virtual_vram_gb:.2f}"))
    logging.info(dash_line)

    # Block Analysis Table
    block_summary = defaultdict(lambda: {'count': 0, 'memory': 0})
    total_memory = 0

    for block in all_blocks:
        block_type = type(block).__name__
        block_memory = sum(p.numel() * p.element_size() for p in block.parameters())
        block_summary[block_type]['count'] += 1
        block_summary[block_type]['memory'] += block_memory
        total_memory += block_memory

    logging.info("            DisTorch SafeTensor Block Analysis")
    logging.info(dash_line)
    fmt_layer = "{:<20}{:>10}{:>15}{:>12}"
    logging.info(fmt_layer.format("Block Type", "Count", "Memory (MB)", "% Total"))
    logging.info(dash_line)
    
    sorted_blocks = sorted(block_summary.items(), key=lambda x: x[1]['memory'], reverse=True)

    for block_type, data in sorted_blocks:
        mem_mb = data['memory'] / (1024 * 1024)
        mem_percent = (data['memory'] / total_memory) * 100 if total_memory > 0 else 0
        logging.info(fmt_layer.format(block_type, str(data['count']), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))
    logging.info(dash_line)

    # Final Assignment Table
    model_size_gb = total_memory / (1024**3)
    block_size_gb = model_size_gb / len(all_blocks) if all_blocks else 0
    blocks_to_offload = int(virtual_vram_gb / block_size_gb) if block_size_gb > 0 else 0
    blocks_on_compute = len(all_blocks) - blocks_to_offload

    logging.info("         DisTorch Final Block Assignments")
    logging.info(dash_line)
    fmt_final = "{:<20}{:>15}"
    logging.info(fmt_final.format("Total Model Size (GB):", f"{model_size_gb:.2f}"))
    logging.info(fmt_final.format("Average Block Size (MB):", f"{block_size_gb * 1024:.2f}" if all_blocks else "N/A"))
    logging.info(dash_line)
    logging.info(fmt_final.format("Blocks on Compute:", f"{blocks_on_compute}"))
    logging.info(fmt_final.format("Blocks on Swap:", f"{blocks_to_offload}"))
    logging.info(eq_line)


def apply_block_swap(model_patcher, compute_device="cuda:0", swap_device="cpu",
                    virtual_vram_gb=4.0, expert_mode_allocations=""):
    """
    Applies block swapping using a manager and PyTorch hooks for robustness.
    """
    logging.info(f"[BlockSwap] Initializing block swap: compute_device={compute_device}, swap_device={swap_device}")
    
    model_to_patch = None
    if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
        model_to_patch = model_patcher.model.diffusion_model
        logging.info("[BlockSwap] Found 'diffusion_model' for patching.")
    elif hasattr(model_patcher, 'model'):
        model_to_patch = model_patcher.model
        logging.info("[BlockSwap] Found 'model' for patching.")
    else:
        logging.error("[BlockSwap] Could not find a valid model to patch.")
        return

    all_blocks = []
    # Block identification logic (remains the same)
    if hasattr(model_to_patch, 'input_blocks') and hasattr(model_to_patch, 'middle_block') and hasattr(model_to_patch, 'output_blocks'):
        logging.info("[BlockSwap] Found standard UNet structure.")
        all_blocks.extend(model_to_patch.input_blocks)
        if isinstance(model_to_patch.middle_block, torch.nn.Module):
            all_blocks.append(model_to_patch.middle_block)
        all_blocks.extend(model_to_patch.output_blocks)
    elif hasattr(model_to_patch, 'blocks') and isinstance(model_to_patch.blocks, torch.nn.ModuleList):
        logging.info("[BlockSwap] Found 'blocks' attribute.")
        all_blocks.extend(model_to_patch.blocks)
    elif hasattr(model_to_patch, 'layers') and isinstance(model_to_patch.layers, torch.nn.ModuleList):
        logging.info("[BlockSwap] Found 'layers' attribute.")
        all_blocks.extend(model_to_patch.layers)
    else:
        logging.info("[BlockSwap] No standard structure found. Searching for top-level ModuleLists.")
        for child in model_to_patch.children():
            if isinstance(child, torch.nn.ModuleList):
                all_blocks.extend(child)

    if not all_blocks:
        logging.error("[BlockSwap] CRITICAL: No swappable blocks found.")
        return
    
    logging.info(f"[BlockSwap] Identified {len(all_blocks)} swappable blocks.")

    # Run analysis before making changes
    analyze_safetensor_distorch(model_to_patch, compute_device, swap_device, virtual_vram_gb, all_blocks)

    # Log initial memory state
    log_memory_usage(compute_device, "Before Swap")
    log_memory_usage(swap_device, "Before Swap")

    model_size_gb = sum(p.numel() * p.element_size() for p in model_to_patch.parameters()) / (1024**3)
    block_size_gb = model_size_gb / len(all_blocks) if all_blocks else 0
    blocks_to_offload_count = int(virtual_vram_gb / block_size_gb) if block_size_gb > 0 else 0
    
    # The blocks at the end of the list are swapped
    blocks_to_swap = all_blocks[-blocks_to_offload_count:] if blocks_to_offload_count > 0 else []

    if not blocks_to_swap:
        logging.warning("[BlockSwap] No blocks designated for swapping based on virtual_vram_gb. Skipping hook setup.")
        return

    # Instantiate and apply the manager
    manager = BlockSwapManager(swap_device=swap_device)
    manager.apply_swap_optimization(blocks_to_swap)

    # Store the manager on the model_patcher for lifecycle management (e.g., cleanup)
    if not hasattr(model_patcher, 'block_swap_managers'):
        model_patcher.block_swap_managers = []
    model_patcher.block_swap_managers.append(manager)
    
    logging.info(f"[BlockSwap] Moved {len(blocks_to_swap)} blocks to {swap_device} and applied hooks.")

    # Log memory state after moving blocks
    log_memory_usage(compute_device, "After Swap")
    log_memory_usage(swap_device, "After Swap")

    logging.info("[BlockSwap] Block swap setup complete.")


def override_class_with_distorch_safetensor(cls):
    """DisTorch 2.0 wrapper for SafeTensor models, providing block-swap memory optimization."""
    from .nodes import get_device_list
    
    class NodeOverrideDisTorchSafeTensorv2(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            compute_device = devices[1] if len(devices) > 1 else devices[0]
            
            inputs["optional"] = inputs.get("optional", {})
            
            inputs["optional"]["compute_device"] = (devices, {"default": compute_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1})
            inputs["optional"]["donor_device"] = (devices, {"default": "cpu"})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"

        def override(self, *args, compute_device=None, virtual_vram_gb=4.0, 
                     donor_device="cpu", expert_mode_allocations="", **kwargs):
            from . import set_current_device
            
            logging.info(f"[DisTorch SafeTensor] Override called with: compute_device={compute_device}, donor_device={donor_device}, virtual_vram_gb={virtual_vram_gb}")

            if compute_device is not None:
                set_current_device(compute_device)
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            model = out[0]
            if hasattr(model, 'model'):
                logging.info("[DisTorch SafeTensor] Model has 'model' attribute, applying block swap.")
                apply_block_swap(
                    model,
                    compute_device=compute_device,
                    swap_device=donor_device,
                    virtual_vram_gb=virtual_vram_gb,
                    expert_mode_allocations=expert_mode_allocations
                )
            else:
                logging.warning("[DisTorch SafeTensor] Loaded object does not have a 'model' attribute, skipping block swap.")
            
            return out

    return NodeOverrideDisTorchSafeTensorv2


# For backwards compatibility, keep the old name pointing to the new safetensor wrapper
override_class_with_distorch_bs = override_class_with_distorch_safetensor
