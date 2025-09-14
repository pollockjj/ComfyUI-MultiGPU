"""
DisTorch Safetensor Memory Management Module
Contains all safetensor related code for distributed memory management
"""

import sys
import torch
import logging
import hashlib
import re
import gc

logger = logging.getLogger("MultiGPU")
import copy
import inspect
from collections import defaultdict
import comfy.model_management as mm
import comfy.model_patcher
from . import current_device
from .device_utils import get_device_list, soft_empty_cache_multigpu

safetensor_allocation_store = {}
safetensor_settings_store = {}


def create_safetensor_model_hash(model, caller):
    """Create a unique hash for a safetensor model to track allocations"""
    if hasattr(model, 'model'):
        # For ModelPatcher objects
        actual_model = model.model
        model_type = type(actual_model).__name__
        # Use ComfyUI's model_size if available
        if hasattr(model, 'model_size'):
            model_size = model.model_size()
        else:
            model_size = sum(p.numel() * p.element_size() for p in actual_model.parameters())
        if hasattr(model, 'model_state_dict'):
            first_layers = str(list(model.model_state_dict().keys())[:3])
        else:
            first_layers = str(list(actual_model.state_dict().keys())[:3])
    else:
        # Direct model
        model_type = type(model).__name__
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        first_layers = str(list(model.state_dict().keys())[:3])
    
    identifier = f"{model_type}_{model_size}_{first_layers}"
    final_hash = hashlib.sha256(identifier.encode()).hexdigest()
    
    # DEBUG STATEMENT - ALWAYS LOG THE HASH
    logger.debug(f"[MultiGPU_DisTorch2] Created hash for {caller}: {final_hash[:8]}...")
    return final_hash


def register_patched_safetensor_modelpatcher():
    """Register and patch the ModelPatcher for distributed safetensor loading"""
    from comfy.model_patcher import wipe_lowvram_weight, move_weight_functions
    # Patch ComfyUI's ModelPatcher
    if not hasattr(comfy.model_patcher.ModelPatcher, '_distorch_patched'):
        original_partially_load = comfy.model_patcher.ModelPatcher.partially_load

        def new_partially_load(self, device_to, extra_memory=0, full_load=False, force_patch_weights=False, **kwargs):
            """Override to use our static device assignments"""
            global safetensor_allocation_store

            debug_hash = create_safetensor_model_hash(self, "partial_load")
            allocations = safetensor_allocation_store.get(debug_hash)

            if not hasattr(self.model, '_distorch_high_precision_loras') or not allocations:

                result = original_partially_load(self, device_to, extra_memory, force_patch_weights)
                if hasattr(self, '_distorch_block_assignments'):
                    del self._distorch_block_assignments
                return result

            if not hasattr(self.model, 'current_weight_patches_uuid'):
                self.model.current_weight_patches_uuid = None

            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)

            if unpatch_weights:
                logger.info(f"[MultiGPU_DisTorch2] Patches changed or forced. Unpatching model.")
                self.unpatch_model(self.offload_device, unpatch_weights=True)

            self.patch_model(load_weights=False)

            mem_counter = 0

            is_clip_model = getattr(self, 'is_clip', False)
            if is_clip_model:
                logger.info(f"[MultiGPU_DisTorch2] Using CLIP-specific allocation for model {debug_hash[:8]} (HEAD PRESERVATION ENABLED)")
                device_assignments = analyze_safetensor_loading_clip(self, allocations)
            else:
                logger.debug(f"[MultiGPU_DisTorch2] Using standard allocation for model {debug_hash[:8]} (UNET/VAE - UNTOUCHED)")
                device_assignments = analyze_safetensor_loading(self, allocations)
            
            model_original_dtype = comfy.utils.weight_dtype(self.model.state_dict())
            high_precision_loras = self.model._distorch_high_precision_loras
            loading = self._load_list()
            loading.sort(reverse=True)
            for module_size, module_name, module_object, params in loading:
                if not unpatch_weights and hasattr(module_object, "comfy_patched_weights") and module_object.comfy_patched_weights == True:
                    block_target_device = device_assignments['block_assignments'].get(module_name, device_to)
                    current_module_device = None
                    try:
                        if any(p.numel() > 0 for p in module_object.parameters(recurse=False)):
                           current_module_device = next(module_object.parameters(recurse=False)).device
                    except StopIteration:
                        pass

                    if current_module_device is not None and str(current_module_device) != str(block_target_device):
                        logger.debug(f"[MultiGPU_DisTorch2] Moving already patched {module_name} to {block_target_device}")
                        module_object.to(block_target_device)

                    mem_counter += module_size
                    continue

                # Step 1: Write block/tensor to compute device first
                module_object.to(device_to)

                # Step 2: Apply LoRa patches while on compute device
                weight_key = "{}.weight".format(module_name)
                bias_key = "{}.bias".format(module_name)

                if weight_key in self.patches:
                    self.patch_weight_to_device(weight_key, device_to=device_to)
                if weight_key in self.weight_wrapper_patches:
                    module_object.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.patches:
                    self.patch_weight_to_device(bias_key, device_to=device_to)
                if bias_key in self.weight_wrapper_patches:
                    module_object.bias_function.extend(self.weight_wrapper_patches[bias_key])

                # Step 3: FP8 casting for CPU storage (if enabled)
                block_target_device = device_assignments['block_assignments'].get(module_name, device_to)
                has_patches = weight_key in self.patches or bias_key in self.patches

                if not high_precision_loras and block_target_device == "cpu" and has_patches and model_original_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    for param_name, param in module_object.named_parameters():
                        if param.dtype.is_floating_point:
                            cast_data = comfy.float.stochastic_rounding(param.data, torch.float8_e4m3fn)
                            new_param = torch.nn.Parameter(cast_data.to(torch.float8_e4m3fn))
                            new_param.requires_grad = param.requires_grad
                            setattr(module_object, param_name, new_param)
                            logger.debug(f"[MultiGPU_DisTorch2] Cast {module_name}.{param_name} to FP8 for CPU storage")

                # Step 4: Move to ultimate destination based on DisTorch assignment
                if block_target_device != device_to:
                    logger.debug(f"[MultiGPU_DisTorch2] Moving {module_name} from {device_to} to {block_target_device}")
                    module_object.to(block_target_device)
                    module_object.comfy_cast_weights = True

                # Mark as patched and update memory counter
                module_object.comfy_patched_weights = True
                mem_counter += module_size

            self.model.current_weight_patches_uuid = self.patches_uuid

            logger.info(f"[MultiGPU_DisTorch2] DisTorch loading completed. Total memory: {mem_counter / (1024 * 1024):.2f}MB")

            return 0

        
        comfy.model_patcher.ModelPatcher.partially_load = new_partially_load
        comfy.model_patcher.ModelPatcher._distorch_patched = True
        logger.info("[MultiGPU_DisTorch2] Successfully patched ModelPatcher.partially_load")


def analyze_safetensor_loading(model_patcher, allocations_string):
    """
    Analyze and distribute safetensor model blocks across devices
    Target for refactor back into one function once stability for CLIP is established.
    """
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = allocations_string
    virtual_vram_gb = 0.0

    distorch_alloc, virtual_vram_str = allocations_string.split('#')

    compute_device = virtual_vram_str.split(';')[0]
    logger.info(f"[MultiGPU_DisTorch2] Compute Device: {compute_device}")

    if not distorch_alloc:
        mode = "fraction"
        logger.info("[MultiGPU_DisTorch2] Expert String Examples:")
        logger.info("  Direct(byte) Mode - cuda:0,500mb;cuda:1,3.0g;cpu,5gb* -> '*' cpu = over/underflow device, put 0.50gb on cuda0, 3.00gb on cuda1, and 5.00gb (or the rest) on cpu")
        logger.info("  Ratio(%) Mode - cuda:0,8%;cuda:1,8%;cpu,4% -> 8:8:4 ratio, put 40% on cuda0, 40% on cuda1, and 20% on cpu")
        distorch_alloc = calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str)

    elif any(c in distorch_alloc.lower() for c in ['g', 'm', 'k', 'b']):
        mode = "byte"
        distorch_alloc = calculate_fraction_from_byte_expert_string(model_patcher, distorch_alloc)
    elif "%" in distorch_alloc:
        mode = "ratio"
        distorch_alloc = calculate_fraction_from_ratio_expert_string(model_patcher, distorch_alloc)
        
    all_devices = get_device_list()
    present_devices = {item.split(',')[0] for item in distorch_alloc.split(';') if ',' in item}
    for device in all_devices:
        if device not in present_devices:
            distorch_alloc += f";{device},0.0"

    logger.info(f"[MultiGPU_DisTorch2] Final Allocation String: {distorch_alloc}")

    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    for allocation in distorch_alloc.split(';'):
        if ',' not in allocation:
            continue
        dev_name, fraction = allocation.split(',')
        fraction = float(fraction)
        total_mem_bytes = mm.get_total_memory(torch.device(dev_name))
        alloc_gb = (total_mem_bytes * fraction) / (1024**3)
        DEVICE_RATIOS_DISTORCH[dev_name] = alloc_gb
        device_table[dev_name] = {
            "fraction": fraction,
            "total_gb": total_mem_bytes / (1024**3),
            "alloc_gb": alloc_gb
        }

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Device Allocations")
    logger.info(eq_line)
    
    fmt_rosetta = "{:<8}{:>9}{:>9}{:>11}{:>10}"
    logger.info(fmt_rosetta.format("Device", "VRAM GB", "Dev %", "Model GB", "Dist %"))
    logger.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))
    
    total_allocated_model_bytes = sum(d["alloc_gb"] * (1024**3) for d in device_table.values())

    for dev in sorted_devices:
        total_dev_gb = device_table[dev]["total_gb"]
        alloc_fraction = device_table[dev]["fraction"]
        alloc_gb = device_table[dev]["alloc_gb"]
        
        dist_ratio_percent = (alloc_gb * (1024**3) / total_allocated_model_bytes) * 100 if total_allocated_model_bytes > 0 else 0

        logger.info(fmt_rosetta.format(
            dev,
            f"{total_dev_gb:.2f}",
            f"{alloc_fraction*100:.1f}%",
            f"{alloc_gb:.2f}",
            f"{dist_ratio_percent:.1f}%"
        ))
    
    logger.info(dash_line)

    block_summary = {}
    block_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    raw_block_list = model_patcher._load_list()

    total_memory = sum(module_size for module_size, _, _, _ in raw_block_list)

    MIN_BLOCK_THRESHOLD = total_memory * 0.0001
    logger.debug(f"[MultiGPU_DisTorch2] Total model memory: {total_memory} bytes")
    logger.debug(f"[MultiGPU_DisTorch2] Tiny block threshold (0.01%): {MIN_BLOCK_THRESHOLD} bytes")

    all_blocks = []
    for module_size, module_name, module_object, params in raw_block_list:
        block_type = type(module_object).__name__
        # Populate summary dictionaries
        block_summary[block_type] = block_summary.get(block_type, 0) + 1
        memory_by_type[block_type] += module_size
        all_blocks.append((module_name, module_object, block_type, module_size))

    block_list = [b for b in all_blocks if b[3] >= MIN_BLOCK_THRESHOLD]
    tiny_block_list = [b for b in all_blocks if b[3] < MIN_BLOCK_THRESHOLD]
    
    logger.debug(f"[MultiGPU_DisTorch2] Total blocks: {len(all_blocks)}")
    logger.debug(f"[MultiGPU_DisTorch2] Distributable blocks: {len(block_list)}")
    logger.debug(f"[MultiGPU_DisTorch2] Tiny blocks (<0.01%): {len(tiny_block_list)}")

    logger.info("    DisTorch2 Model Layer Distribution")
    logger.info(dash_line)
    fmt_layer = "{:<18}{:>7}{:>14}{:>10}"
    logger.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    
    for layer_type, count in block_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_layer.format(layer_type[:18], str(count), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))
    
    logger.info(dash_line)

    # Distribute blocks sequentially from the tail of the model

    device_assignments = {device: [] for device in DEVICE_RATIOS_DISTORCH.keys()}
    block_assignments = {}

    # Create a memory quota for each donor device based on its calculated allocation.
    donor_devices = [d for d in sorted_devices]
    donor_quotas = {
        dev: device_table[dev]["alloc_gb"] * (1024**3)
        for dev in donor_devices
    }

    # Iterate from the TAIL of the model, assigning blocks to donors until their quotas are filled.
    for block_name, module, block_type, block_memory in reversed(block_list):
        assigned_to_donor = False
        for donor in donor_devices:
            if donor_quotas[donor] >= block_memory:
                block_assignments[block_name] = donor
                donor_quotas[donor] -= block_memory
                assigned_to_donor = True
                break # Move to the next block
        
        if not assigned_to_donor:  #Note - small rounding errors and tensor-fitting on devices make a block occasionally an orphan. We treat orphans the same as tiny_block_list as they are generally small rounding errors
            block_assignments[block_name] = compute_device

    if tiny_block_list:
        for block_name, module, block_type, block_memory in tiny_block_list:
            block_assignments[block_name] = compute_device

    # Populate device_assignments from the final block_assignments
    for block_name, device in block_assignments.items():
        # Find the block in the original list to get all its info
        for b_name, b_module, b_type, b_mem in all_blocks:
            if b_name == block_name:
                device_assignments[device].append((b_name, b_module, b_type, b_mem))
                break

    logger.info("DisTorch2 Model Final Device/Layer Assignments")
    logger.info(dash_line)
    logger.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    
    if tiny_block_list:
        tiny_block_memory = sum(b[3] for b in tiny_block_list)
        tiny_mem_mb = tiny_block_memory / (1024 * 1024)
        tiny_mem_percent = (tiny_block_memory / total_memory) * 100 if total_memory > 0 else 0
        device_label = f"{compute_device} (<0.01%)"
        logger.info(fmt_assign.format(device_label, str(len(tiny_block_list)), f"{tiny_mem_mb:.2f}", f"{tiny_mem_percent:.1f}%"))
        logger.debug(f"[MultiGPU_DisTorch2] Tiny block memory breakdown: {tiny_block_memory} bytes ({tiny_mem_mb:.2f} MB), which is {tiny_mem_percent:.4f}% of total model memory.")

    total_assigned_memory = 0
    device_memories = {}
    
    for device, blocks in device_assignments.items():
        dist_blocks = [b for b in blocks if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks:
            continue

        device_memory = sum(b[3] for b in dist_blocks)
        device_memories[device] = device_memory
        total_assigned_memory += device_memory

    sorted_assignments = sorted(device_memories.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_assignments:
        # Get only the distributed blocks for the count
        dist_blocks = [b for b in device_assignments[dev] if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks:
            continue
            
        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_assign.format(dev, str(len(dist_blocks)), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))
    
    logger.info(dash_line)

    return {
        "device_assignments": device_assignments,
        "block_assignments": block_assignments
    }


def analyze_safetensor_loading_clip(model_patcher, allocations_string):
    """
    CLIP-SPECIFIC: A 1:1 clone of the working UNET allocation logic with the
    single required modification to preserve head-blocks on the compute device.
    All other logic and UX (logging, etc.) is identical to the original.
    Target for refactor once stability for CLIP is established.
    """
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = allocations_string
    virtual_vram_gb = 0.0

    distorch_alloc, virtual_vram_str = allocations_string.split('#')

    compute_device = virtual_vram_str.split(';')[0]

    logger.info(f"[MultiGPU_DisTorch2_CLIP] CLIP Compute Device: {compute_device}")

    if not distorch_alloc:
        mode = "fraction"
        logger.info("[MultiGPU_DisTorch2_CLIP] Expert String Examples:")
        logger.info("  Direct(byte) Mode - cuda:0,500mb;cuda:1,3.0g;cpu,5gb* -> '*' cpu = over/underflow device, put 0.50gb on cuda0, 3.00gb on cuda1, and 5.00gb (or the rest) on cpu")
        logger.info("  Ratio(%) Mode - cuda:0,8%;cuda:1,8%;cpu,4% -> 8:8:4 ratio, put 40% on cuda0, 40% on cuda1, and 20% on cpu")
        distorch_alloc = calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str)

    elif any(c in distorch_alloc.lower() for c in ['g', 'm', 'k', 'b']):
        mode = "byte"
        distorch_alloc = calculate_fraction_from_byte_expert_string(model_patcher, distorch_alloc)
    elif "%" in distorch_alloc:
        mode = "ratio"
        distorch_alloc = calculate_fraction_from_ratio_expert_string(model_patcher, distorch_alloc)
        
    all_devices = get_device_list()
    present_devices = {item.split(',')[0] for item in distorch_alloc.split(';') if ',' in item}
    for device in all_devices:
        if device not in present_devices:
            distorch_alloc += f";{device},0.0"

    logger.info(f"[MultiGPU_DisTorch2_CLIP] Final CLIP Allocation String: {distorch_alloc}")

    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    for allocation in distorch_alloc.split(';'):
        if ',' not in allocation:
            continue
        dev_name, fraction = allocation.split(',')
        fraction = float(fraction)
        total_mem_bytes = mm.get_total_memory(torch.device(dev_name))
        alloc_gb = (total_mem_bytes * fraction) / (1024**3)
        DEVICE_RATIOS_DISTORCH[dev_name] = alloc_gb
        device_table[dev_name] = {
            "fraction": fraction,
            "total_gb": total_mem_bytes / (1024**3),
            "alloc_gb": alloc_gb
        }

    logger.info(eq_line)
    logger.info("    DisTorch2 CLIP Model Device Allocations")
    logger.info(eq_line)
    
    fmt_rosetta = "{:<8}{:>9}{:>9}{:>11}{:>10}"
    logger.info(fmt_rosetta.format("Device", "VRAM GB", "Dev %", "Model GB", "Dist %"))
    logger.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))
    
    total_allocated_model_bytes = sum(d["alloc_gb"] * (1024**3) for d in device_table.values())

    for dev in sorted_devices:
        total_dev_gb = device_table[dev]["total_gb"]
        alloc_fraction = device_table[dev]["fraction"]
        alloc_gb = device_table[dev]["alloc_gb"]
        
        dist_ratio_percent = (alloc_gb * (1024**3) / total_allocated_model_bytes) * 100 if total_allocated_model_bytes > 0 else 0

        logger.info(fmt_rosetta.format(
            dev,
            f"{total_dev_gb:.2f}",
            f"{alloc_fraction*100:.1f}%",
            f"{alloc_gb:.2f}",
            f"{dist_ratio_percent:.1f}%"
        ))
    
    logger.info(dash_line)

    block_summary = {}
    memory_by_type = defaultdict(int)

    raw_block_list = model_patcher._load_list()
    total_memory = sum(module_size for module_size, _, _, _ in raw_block_list)

    # Split the model into head and distributable parts
    head_keywords = ['embed', 'wte', 'wpe', 'token_embedding', 'position_embedding']
    head_blocks = []
    distributable_blocks_raw = []
    head_memory = 0

    for module_size, module_name, module_object, params in raw_block_list:
        if any(keyword in module_name.lower() for keyword in head_keywords):
            head_blocks.append((module_size, module_name, module_object, params))
        else:
            distributable_blocks_raw.append((module_size, module_name, module_object, params))

    MIN_BLOCK_THRESHOLD = total_memory * 0.0001
    all_blocks = []

    for module_size, module_name, module_object, params in raw_block_list:
        block_type = type(module_object).__name__
        block_summary[block_type] = block_summary.get(block_type, 0) + 1
        memory_by_type[block_type] += module_size
        all_blocks.append((module_name, module_object, block_type, module_size))

    # Use the distributable part for actual allocation logic
    distributable_all_blocks = []
    for module_size, module_name, module_object, params in distributable_blocks_raw:
        distributable_all_blocks.append((module_name, module_object, type(module_object).__name__, module_size))

    block_list = [b for b in distributable_all_blocks if b[3] >= MIN_BLOCK_THRESHOLD]
    tiny_block_list = [b for b in distributable_all_blocks if b[3] < MIN_BLOCK_THRESHOLD]
    
    logger.info("    DisTorch2 CLIP Model Layer Distribution")
    logger.info(dash_line)
    fmt_layer = "{:<18}{:>7}{:>14}{:>10}"
    logger.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    
    for layer_type, count in block_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_layer.format(layer_type[:18], str(count), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))
    
    logger.info(dash_line)

    block_assignments = {}

    # Pre-assign head blocks and calculate their memory usage
    for module_size, module_name, module_object, params in head_blocks:
        block_assignments[module_name] = compute_device
        head_memory += module_size
    if head_blocks:
        logger.info(f"[MultiGPU_DisTorch2_CLIP] Preserving {len(head_blocks)} head layer(s) ({head_memory / (1024*1024):.2f} MB) on compute device: {compute_device}")
    donor_devices = [d for d in sorted_devices]
    donor_quotas = {
        dev: device_table[dev]["alloc_gb"] * (1024**3)
        for dev in donor_devices
    }
    # Adjust compute_device quota to account for the locked head
    if compute_device in donor_quotas:
        donor_quotas[compute_device] = max(0, donor_quotas[compute_device] - head_memory)

    for block_name, module, block_type, block_memory in reversed(block_list):
        assigned_to_donor = False
        for donor in donor_devices:
            if donor_quotas[donor] >= block_memory:
                block_assignments[block_name] = donor
                donor_quotas[donor] -= block_memory
                assigned_to_donor = True
                break # Move to the next block
        
        if not assigned_to_donor:
            block_assignments[block_name] = compute_device

    for block_name, module, block_type, block_memory in tiny_block_list:
        block_assignments[block_name] = compute_device

    device_assignments = {device: [] for device in DEVICE_RATIOS_DISTORCH.keys()}
    for block_name, device in block_assignments.items():
        # Find the block in the original list to get all its info
        for b_name, b_module, b_type, b_mem in all_blocks:
            if b_name == block_name:
                device_assignments[device].append((b_name, b_module, b_type, b_mem))
                break

    logger.info("DisTorch2 CLIP Model Final Device/Layer Assignments")
    logger.info(dash_line)
    logger.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    
    device_memories = defaultdict(int)
    device_counts = defaultdict(int)
    for device, blocks in device_assignments.items():
        for b_name, b_module, b_type, b_mem in blocks:
            device_memories[device] += b_mem
            device_counts[device] += 1

    sorted_assignments = sorted(device_memories.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_assignments:
        if device_counts[dev] == 0:
            continue
        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_assign.format(dev, str(device_counts[dev]), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))
    
    logger.info(dash_line)

    return {
        "device_assignments": device_assignments,
        "block_assignments": block_assignments
    }


def parse_memory_string(mem_str):
    """Parses a memory string (e.g., '4.0g', '512M') and returns bytes."""
    mem_str = mem_str.strip().lower()
    match = re.match(r'(\d+\.?\d*)\s*([gmkb]?)', mem_str)
    if not match:
        raise ValueError(f"Invalid memory string format: {mem_str}")
    
    val, unit = match.groups()
    val = float(val)
    
    if unit == 'g':
        return val * (1024**3)
    elif unit == 'm':
        return val * (1024**2)
    elif unit == 'k':
        return val * 1024
    else: # b or no unit
        return val

def calculate_fraction_from_byte_expert_string(model_patcher, byte_str):
    """
    Converts a user-provided byte string (e.g., "cuda:1,4gb;cpu,*") into a
    fractional VRAM allocation string that the main assignment logic can use.
    This function strictly respects device order and byte quotas.
    """
    raw_block_list = model_patcher._load_list()
    total_model_memory = sum(module_size for module_size, _, _, _ in raw_block_list)
    remaining_model_bytes = total_model_memory

    # Use a list of tuples to preserve the user-defined order
    parsed_allocations = []
    wildcard_device = "cpu"  # Default wildcard device

    for allocation in byte_str.split(';'):
        if ',' not in allocation:
            continue
        dev_name, val_str = allocation.split(',', 1)
        is_wildcard = '*' in val_str
        
        if is_wildcard:
            wildcard_device = dev_name
            # Don't add wildcard to the priority list yet
        else:
            byte_val = parse_memory_string(val_str)
            parsed_allocations.append({'device': dev_name, 'bytes': byte_val})

    final_byte_allocations = defaultdict(int)

    # Process devices with specific byte allocations first, in order
    for alloc in parsed_allocations:
        dev = alloc['device']
        requested_bytes = alloc['bytes']

        # Determine the actual bytes to allocate to this device
        bytes_to_assign = min(requested_bytes, remaining_model_bytes)
        
        if bytes_to_assign > 0:
            final_byte_allocations[dev] = bytes_to_assign
            remaining_model_bytes -= bytes_to_assign
            logger.info(f"[MultiGPU_DisTorch2] Assigning {bytes_to_assign / (1024**2):.2f}MB of model to {dev} (requested {requested_bytes / (1024**2):.2f}MB).")
        
        if remaining_model_bytes <= 0:
            logger.info("[MultiGPU_DisTorch2] All model blocks have been allocated. Subsequent devices in the string will receive no assignment.")
            break

    # Assign any leftover model bytes to the wildcard device
    if remaining_model_bytes > 0:
        final_byte_allocations[wildcard_device] += remaining_model_bytes
        logger.info(f"[MultiGPU_DisTorch2] Assigning remaining {remaining_model_bytes / (1024**2):.2f}MB of model to wildcard device '{wildcard_device}'.")

    # Convert the final byte allocations to VRAM fractions
    allocation_parts = []
    for dev, bytes_alloc in final_byte_allocations.items():
        total_device_vram = mm.get_total_memory(torch.device(dev))
        if total_device_vram > 0:
            fraction = bytes_alloc / total_device_vram
            allocation_parts.append(f"{dev},{fraction:.4f}")
    
    allocations_string = ";".join(allocation_parts)

    return allocations_string

def calculate_fraction_from_ratio_expert_string(model_patcher, ratio_str):
    """
    Converts a user-provided ratio string (which describes how to split the MODEL)
    into a fraction string (which describes the fraction of DEVICE VRAM to use).
    """
    raw_block_list = model_patcher._load_list()
    total_model_memory = sum(module_size for module_size, _, _, _ in raw_block_list)

    raw_ratios = {}
    for allocation in ratio_str.split(';'):
        if ',' not in allocation: continue
        dev_name, val_str = allocation.split(',', 1)
        # Assumes the value is a unitless ratio number, ignores '%' for simplicity.
        value = float(val_str.replace('%','').strip())
        raw_ratios[dev_name] = value

    total_ratio_parts = sum(raw_ratios.values())
    allocation_parts = []

    for dev, ratio_val in raw_ratios.items():
        bytes_of_model_for_device = (ratio_val / total_ratio_parts) * total_model_memory

        total_vram_of_device = mm.get_total_memory(torch.device(dev))

        if total_vram_of_device > 0:
            required_fraction = bytes_of_model_for_device / total_vram_of_device
            allocation_parts.append(f"{dev},{required_fraction:.4f}")

    ratio_values = [str(v) for v in raw_ratios.values()]
    ratio_string = ":".join(ratio_values)

    normalized_pcts = [(v / total_ratio_parts) * 100 for v in raw_ratios.values()]
    
    put_parts = []
    for i, dev_name in enumerate(raw_ratios.keys()):
        put_parts.append(f"{int(normalized_pcts[i])}% on {dev_name}")

    if len(put_parts) == 1:
        put_part = put_parts[0]
    elif len(put_parts) == 2:
        put_part = f"{put_parts[0]} and {put_parts[1]}"
    else:
        put_part = ", ".join(put_parts[:-1]) + f", and {put_parts[-1]}"
    
    logger.info(f"[MultiGPU_DisTorch2] Ratio(%) Mode - {ratio_str} -> {ratio_string} ratio, put {put_part}")

    allocations_string = ";".join(allocation_parts)

    return allocations_string

def calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str):
    """Calculate virtual VRAM allocation string for distributed safetensor loading"""
    recipient_device, vram_amount, donors = virtual_vram_str.split(';')
    virtual_vram_gb = float(vram_amount)

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<8} {:<6} {:>11} {:>9} {:>9}"

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Virtual VRAM Analysis")
    logger.info(eq_line)
    logger.info(fmt_assign.format("Object", "Role", "Original(GB)", "Total(GB)", "Virt(GB)"))
    logger.info(dash_line)

    # Calculate recipient VRAM
    recipient_vram = mm.get_total_memory(torch.device(recipient_device)) / (1024**3)
    recipient_virtual = recipient_vram + virtual_vram_gb

    logger.info(fmt_assign.format(recipient_device, 'recip', f"{recipient_vram:.2f}GB",f"{recipient_virtual:.2f}GB", f"+{virtual_vram_gb:.2f}GB"))

    # Handle donor devices
    ram_donors = [d for d in donors.split(',')]
    remaining_vram_needed = virtual_vram_gb
    
    donor_device_info = {}
    donor_allocations = {}
    
    for donor in ram_donors:
        donor_vram = mm.get_total_memory(torch.device(donor)) / (1024**3)
        max_donor_capacity = donor_vram
        
        donation = min(remaining_vram_needed, max_donor_capacity)
        donor_virtual = donor_vram - donation
        remaining_vram_needed -= donation
        donor_allocations[donor] = donation
            
        donor_device_info[donor] = (donor_vram, donor_virtual)
        logger.info(fmt_assign.format(donor, 'donor', f"{donor_vram:.2f}GB",  f"{donor_virtual:.2f}GB", f"-{donation:.2f}GB"))
    
    
    logger.info(dash_line)

    # Calculate model size
    model = model_patcher.model if hasattr(model_patcher, 'model') else model_patcher
    total_memory = 0
    
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if module.weight is not None:
                total_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                total_memory += module.bias.numel() * module.bias.element_size()

    model_size_gb = total_memory / (1024**3)
    new_model_size_gb = max(0, model_size_gb - virtual_vram_gb)

    logger.info(fmt_assign.format('model', 'model', f"{model_size_gb:.2f}GB",f"{new_model_size_gb:.2f}GB", f"-{virtual_vram_gb:.2f}GB"))

    # Warning if model too large
    if model_size_gb > (recipient_vram * 0.9):
        required_offload_gb = model_size_gb - (recipient_vram * 0.9)
        logger.warning(f"[MultiGPU] WARNING: Model size ({model_size_gb:.2f}GB) is larger than 90% of available VRAM on {recipient_device} ({recipient_vram * 0.9:.2f}GB).")
        logger.warning(f"[MultiGPU] To prevent an OOM error, set 'virtual_vram_gb' to at least {required_offload_gb:.2f}.")

    new_on_recipient = max(0, model_size_gb - virtual_vram_gb)

    # Build allocation string
    allocation_parts = []
    recipient_percent = new_on_recipient / recipient_vram
    allocation_parts.append(f"{recipient_device},{recipient_percent:.4f}")

    for donor in ram_donors:
        donor_vram = donor_device_info[donor][0]
        donor_percent = donor_allocations[donor] / donor_vram
        allocation_parts.append(f"{donor},{donor_percent:.4f}")
    
    allocations_string = ";".join(allocation_parts)
    return allocations_string

def override_class_with_distorch_safetensor_v2(cls):
    """DisTorch 2.0 wrapper for safetensor models"""
    from . import current_device

    class NodeOverrideDisTorchSafetensorV2(cls):
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
            inputs["optional"]["high_precision_loras"] = ("BOOLEAN", {"default": True})
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch2)"

        @classmethod
        def IS_CHANGED(s, *args, compute_device=None, virtual_vram_gb=4.0,
                       donor_device="cpu", expert_mode_allocations="", high_precision_loras=True, **kwargs):
            # Create a hash of our specific settings
            settings_str = f"{compute_device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}{high_precision_loras}"
            return hashlib.sha256(settings_str.encode()).hexdigest()

        def override(self, *args, compute_device=None, virtual_vram_gb=4.0,
                     donor_device="cpu", expert_mode_allocations="", high_precision_loras=True, **kwargs):

            from . import set_current_device
            if compute_device is not None:
                set_current_device(compute_device)

            # Register our patched ModelPatcher
            register_patched_safetensor_modelpatcher()

            # Call original function
            fn = getattr(super(), cls.FUNCTION)

            # --- Check if we need to unload the model due to settings change ---
            # This logic is a bit redundant with IS_CHANGED, but provides clear logging
            settings_str = f"{compute_device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}"
            settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()

            # Temporarily load to get hash without applying our patch
            temp_out = fn(*args, **kwargs)
            model_to_check = None
            if hasattr(temp_out[0], 'model'):
                model_to_check = temp_out[0]
            elif hasattr(temp_out[0], 'patcher') and hasattr(temp_out[0].patcher, 'model'):
                model_to_check = temp_out[0].patcher

            if model_to_check:
                model_hash = create_safetensor_model_hash(model_to_check, "override_check")
                last_settings_hash = safetensor_settings_store.get(model_hash)

                if last_settings_hash != settings_hash:
                    logger.info(f"[MultiGPU_DisTorch2] Settings changed for model {model_hash[:8]}. Previous settings hash: {last_settings_hash}, New settings hash: {settings_hash}. Forcing reload.")
                else:
                    logger.info(f"[MultiGPU_DisTorch2] Settings unchanged for model {model_hash[:8]}. Using cached model.")

            out = fn(*args, **kwargs)

            # Store high_precision_loras in the model for later retrieval
            if hasattr(out[0], 'model'):
                out[0].model._distorch_high_precision_loras = high_precision_loras
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                out[0].patcher.model._distorch_high_precision_loras = high_precision_loras

            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{compute_device};{virtual_vram_gb};{donor_device}"
            elif expert_mode_allocations:  # Only include compute device if there's an expert string
                vram_string = compute_device

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""

            logger.info(f"[MultiGPU_DisTorch2] Full allocation string: {full_allocation}")

            if hasattr(out[0], 'model'):
                model_hash = create_safetensor_model_hash(out[0], "override")
                safetensor_allocation_store[model_hash] = full_allocation
                safetensor_settings_store[model_hash] = settings_hash
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_safetensor_model_hash(out[0].patcher, "override")
                safetensor_allocation_store[model_hash] = full_allocation
                safetensor_settings_store[model_hash] = settings_hash

            return out

    return NodeOverrideDisTorchSafetensorV2


def override_class_with_distorch_safetensor_v2_clip(cls):
    """DisTorch 2.0 wrapper for safetensor CLIP models"""
    from . import current_device

    class NodeOverrideDisTorchSafetensorV2Clip(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]

            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})  # Changed from compute_device
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1})
            inputs["optional"]["donor_device"] = (devices, {"default": "cpu"})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            inputs["optional"]["high_precision_loras"] = ("BOOLEAN", {"default": True})
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch2)"

        @classmethod
        def IS_CHANGED(s, *args, device=None, virtual_vram_gb=4.0,  # Changed from compute_device
                       donor_device="cpu", expert_mode_allocations="", high_precision_loras=True, **kwargs):
            # Create a hash of our specific settings
            settings_str = f"{device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}{high_precision_loras}"  # Changed from compute_device
            return hashlib.sha256(settings_str.encode()).hexdigest()

        def override(self, *args, device=None, virtual_vram_gb=4.0,  # Changed from compute_device
                     donor_device="cpu", expert_mode_allocations="", high_precision_loras=True, **kwargs):

            from . import set_current_text_encoder_device  # Use text encoder device setter
            if device is not None:
                set_current_text_encoder_device(device)

            kwargs['device'] = 'default'  # Hardcode device setting like in standard clip wrapper

            # Register our patched ModelPatcher
            register_patched_safetensor_modelpatcher()

            # Call original function
            fn = getattr(super(), cls.FUNCTION)

            # --- Check if we need to unload the model due to settings change ---
            # This logic is a bit redundant with IS_CHANGED, but provides clear logging
            settings_str = f"{device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}"  # Changed from compute_device
            settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()

            # Temporarily load to get hash without applying our patch
            temp_out = fn(*args, **kwargs)
            model_to_check = None
            if hasattr(temp_out[0], 'model'):
                model_to_check = temp_out[0]
            elif hasattr(temp_out[0], 'patcher') and hasattr(temp_out[0].patcher, 'model'):
                model_to_check = temp_out[0].patcher

            if model_to_check:
                model_hash = create_safetensor_model_hash(model_to_check, "override_check")
                last_settings_hash = safetensor_settings_store.get(model_hash)

                if last_settings_hash != settings_hash:
                    logger.info(f"[MultiGPU_DisTorch2] Settings changed for model {model_hash[:8]}. Previous settings hash: {last_settings_hash}, New settings hash: {settings_hash}. Forcing reload.")
                else:
                    logger.info(f"[MultiGPU_DisTorch2] Settings unchanged for model {model_hash[:8]}. Using cached model.")

            out = fn(*args, **kwargs)

            # Store high_precision_loras in the model for later retrieval
            if hasattr(out[0], 'model'):
                out[0].model._distorch_high_precision_loras = high_precision_loras
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                out[0].patcher.model._distorch_high_precision_loras = high_precision_loras

            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{device};{virtual_vram_gb};{donor_device}"  # Changed from compute_device
            elif expert_mode_allocations:  # Only include device if there's an expert string
                vram_string = device  # Changed from compute_device

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""

            logger.info(f"[MultiGPU_DisTorch2] Full allocation string: {full_allocation}")

            if hasattr(out[0], 'model'):
                model_hash = create_safetensor_model_hash(out[0], "override")
                safetensor_allocation_store[model_hash] = full_allocation
                safetensor_settings_store[model_hash] = settings_hash
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_safetensor_model_hash(out[0].patcher, "override")
                safetensor_allocation_store[model_hash] = full_allocation
                safetensor_settings_store[model_hash] = settings_hash

            return out

    return NodeOverrideDisTorchSafetensorV2Clip

def override_class_with_distorch_safetensor_v2_clip_no_device(cls):
    """DisTorch 2.0 wrapper for safetensor CLIP models"""
    from . import current_device

    class NodeOverrideDisTorchSafetensorV2ClipNoDevice(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]

            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})  # Changed from compute_device
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1})
            inputs["optional"]["donor_device"] = (devices, {"default": "cpu"})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            inputs["optional"]["high_precision_loras"] = ("BOOLEAN", {"default": True})
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch2)"

        @classmethod
        def IS_CHANGED(s, *args, device=None, virtual_vram_gb=4.0,  # Changed from compute_device
                       donor_device="cpu", expert_mode_allocations="", high_precision_loras=True, **kwargs):
            # Create a hash of our specific settings
            settings_str = f"{device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}{high_precision_loras}"  # Changed from compute_device
            return hashlib.sha256(settings_str.encode()).hexdigest()

        def override(self, *args, device=None, virtual_vram_gb=4.0,  # Changed from compute_device
                     donor_device="cpu", expert_mode_allocations="", high_precision_loras=True, **kwargs):

            from . import set_current_text_encoder_device  # Use text encoder device setter
            if device is not None:
                set_current_text_encoder_device(device)

            # Register our patched ModelPatcher
            register_patched_safetensor_modelpatcher()

            # Call original function
            fn = getattr(super(), cls.FUNCTION)

            # --- Check if we need to unload the model due to settings change ---
            # This logic is a bit redundant with IS_CHANGED, but provides clear logging
            settings_str = f"{device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}"  # Changed from compute_device
            settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()

            # Temporarily load to get hash without applying our patch
            temp_out = fn(*args, **kwargs)
            model_to_check = None
            if hasattr(temp_out[0], 'model'):
                model_to_check = temp_out[0]
            elif hasattr(temp_out[0], 'patcher') and hasattr(temp_out[0].patcher, 'model'):
                model_to_check = temp_out[0].patcher

            if model_to_check:
                model_hash = create_safetensor_model_hash(model_to_check, "override_check")
                last_settings_hash = safetensor_settings_store.get(model_hash)

                if last_settings_hash != settings_hash:
                    logger.info(f"[MultiGPU_DisTorch2] Settings changed for model {model_hash[:8]}. Previous settings hash: {last_settings_hash}, New settings hash: {settings_hash}. Forcing reload.")
                else:
                    logger.info(f"[MultiGPU_DisTorch2] Settings unchanged for model {model_hash[:8]}. Using cached model.")

            out = fn(*args, **kwargs)

            # Store high_precision_loras in the model for later retrieval
            if hasattr(out[0], 'model'):
                out[0].model._distorch_high_precision_loras = high_precision_loras
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                out[0].patcher.model._distorch_high_precision_loras = high_precision_loras

            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{device};{virtual_vram_gb};{donor_device}"  # Changed from compute_device
            elif expert_mode_allocations:  # Only include device if there's an expert string
                vram_string = device  # Changed from compute_device

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""

            logger.info(f"[MultiGPU_DisTorch2] Full allocation string: {full_allocation}")

            if hasattr(out[0], 'model'):
                model_hash = create_safetensor_model_hash(out[0], "override")
                safetensor_allocation_store[model_hash] = full_allocation
                safetensor_settings_store[model_hash] = settings_hash
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_safetensor_model_hash(out[0].patcher, "override")
                safetensor_allocation_store[model_hash] = full_allocation
                safetensor_settings_store[model_hash] = settings_hash

            return out

    return NodeOverrideDisTorchSafetensorV2ClipNoDevice
