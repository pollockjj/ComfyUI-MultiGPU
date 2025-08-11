"""
Block Swap Module for SafeTensor Models
Contains all SafeTensor DisTorch code for block-swap memory optimization
"""

import torch
import logging
import copy
from collections import defaultdict
import comfy.model_management as mm


def analyze_safetensor_distorch(model, compute_device, swap_device, virtual_vram_gb, reserved_swap_gb, all_blocks):
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
    
    logging.info(fmt_assign.format(compute_device, "Compute", f"{compute_total_gb:.2f}", f"Reserve: {reserved_swap_gb:.2f}"))
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
    Applies WanVideo-style block swapping by patching the forward method of individual model blocks.
    This allows for offloading parts of the model to a swap device to conserve VRAM.
    """
    logging.info(f"[DisTorch SafeTensor] Initializing block swap: compute_device={compute_device}, swap_device={swap_device}")
    
    model_to_patch = None
    if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
        model_to_patch = model_patcher.model.diffusion_model
        logging.info("[DisTorch SafeTensor] Found 'diffusion_model' attribute for patching.")
    elif hasattr(model_patcher, 'model'):
        model_to_patch = model_patcher.model
        logging.info("[DisTorch SafeTensor] Found 'model' attribute for patching.")
    else:
        logging.error("[DisTorch SafeTensor] Could not find a valid model to patch for block swapping.")
        return

    all_blocks = []
    # 1. Standard UNet Structure
    if hasattr(model_to_patch, 'input_blocks') and hasattr(model_to_patch, 'middle_block') and hasattr(model_to_patch, 'output_blocks'):
        logging.info("[DisTorch SafeTensor] Found standard UNet structure ('input_blocks', 'middle_block', 'output_blocks').")
        all_blocks.extend(model_to_patch.input_blocks)
        if isinstance(model_to_patch.middle_block, torch.nn.Module):
            all_blocks.append(model_to_patch.middle_block)
        all_blocks.extend(model_to_patch.output_blocks)
    # 2. Simple 'blocks' attribute
    elif hasattr(model_to_patch, 'blocks') and isinstance(model_to_patch.blocks, torch.nn.ModuleList):
        logging.info("[DisTorch SafeTensor] Found 'blocks' attribute of type ModuleList.")
        all_blocks.extend(model_to_patch.blocks)
    # 3. Simple 'layers' attribute
    elif hasattr(model_to_patch, 'layers') and isinstance(model_to_patch.layers, torch.nn.ModuleList):
        logging.info("[DisTorch SafeTensor] Found 'layers' attribute of type ModuleList.")
        all_blocks.extend(model_to_patch.layers)
    # 4. Fallback to top-level ModuleLists
    else:
        logging.info("[DisTorch SafeTensor] No standard structure found. Falling back to searching for top-level ModuleLists.")
        for child in model_to_patch.children():
            if isinstance(child, torch.nn.ModuleList):
                logging.info(f"[DisTorch SafeTensor] Found top-level ModuleList with {len(child)} modules. Adding them as blocks.")
                all_blocks.extend(child)

    if not all_blocks:
        logging.error("[DisTorch SafeTensor] CRITICAL: No swappable blocks were found in the model. Block swap cannot be applied.")
        return
    
    logging.info(f"[DisTorch SafeTensor] Successfully identified {len(all_blocks)} swappable blocks.")

    # Run and display the analysis
    analyze_safetensor_distorch(model_to_patch, compute_device, swap_device, virtual_vram_gb, 0.0, all_blocks)

    model_size_gb = sum(p.numel() * p.element_size() for p in model_to_patch.parameters()) / (1024**3)
    block_size_gb = model_size_gb / len(all_blocks) if all_blocks else 0
    blocks_to_offload = int(virtual_vram_gb / block_size_gb) if block_size_gb > 0 else 0
    blocks_on_compute = len(all_blocks) - blocks_to_offload

    for i, block in enumerate(all_blocks):
        # Determine target device for this block
        target_device = compute_device if i < blocks_on_compute else swap_device
        block.to(target_device)

        # Patch the forward method only if the block is on the swap device
        if target_device == swap_device:
            original_forward = block.forward
            
            def create_patched_forward(original_f, b, block_index, cd, sd):
                def patched_forward(*args, **kwargs):
                    logging.info(f"[DisTorch SafeTensor] Swapping block {block_index} to {cd} for computation.")
                    b.to(cd, non_blocking=True)
                    result = original_f(*args, **kwargs)
                    logging.info(f"[DisTorch SafeTensor] Swapping block {block_index} back to {sd}.")
                    b.to(sd, non_blocking=True)
                    return result
                return patched_forward

            block.forward = create_patched_forward(original_forward, block, i, torch.device(compute_device), torch.device(swap_device))
            logging.info(f"[DisTorch SafeTensor] Patched forward method for block {i} on {swap_device}.")

    logging.info("[DisTorch SafeTensor] Block swap setup complete.")


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
