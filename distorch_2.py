"""
DisTorch Safetensor Memory Management Module
Contains all safetensor related code for distributed memory management
"""

import sys
import torch
import logging
import hashlib
import re

logger = logging.getLogger("MultiGPU")
import copy
import inspect
from collections import defaultdict
import comfy.model_management as mm
import comfy.model_patcher
from . import current_device

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

            mem_counter = 0

            logger.info(f"[MultiGPU_DisTorch2] Using static allocation for model {debug_hash[:8]}")
            device_assignments = analyze_safetensor_loading(self, allocations)
            model_original_dtype = comfy.utils.weight_dtype(self.model.state_dict())
            high_precision_loras = self.model._distorch_high_precision_loras
            loading = self._load_list()
            loading.sort(reverse=True)
            for module_size, module_name, module_object, params in loading:
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

            logger.info(f"[MultiGPU_DisTorch2] DisTorch loading completed. Total memory: {mem_counter / (1024 * 1024):.2f}MB")

            return 0

        
        comfy.model_patcher.ModelPatcher.partially_load = new_partially_load
        comfy.model_patcher.ModelPatcher._distorch_patched = True
        logger.info("[MultiGPU_DisTorch2] Successfully patched ModelPatcher.partially_load")


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

def analyze_safetensor_loading(model_patcher, allocations_str):
    """
    Analyze and distribute safetensor model blocks across devices
    """
    distorch_alloc = allocations_str
    if '#' in allocations_str:
        distorch_alloc, _ = allocations_str.split('#', 1)

    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    raw_block_list = model_patcher._load_list()
    total_memory = sum(module_size for module_size, _, _, _ in raw_block_list)

    # Segregate tiny blocks and recalculate total memory for precision
    block_summary = {}
    memory_by_type = defaultdict(int)
    MIN_BLOCK_THRESHOLD = total_memory * 0.0001
    
    all_blocks = []
    for module_size, module_name, module_object, params in raw_block_list:
        block_type = type(module_object).__name__
        block_summary[block_type] = block_summary.get(block_type, 0) + 1
        memory_by_type[block_type] += module_size
        all_blocks.append((module_name, module_object, block_type, module_size))

    block_list = [b for b in all_blocks if b[3] >= MIN_BLOCK_THRESHOLD]
    tiny_block_list = [b for b in all_blocks if b[3] < MIN_BLOCK_THRESHOLD]
    
    tiny_block_memory = sum(b[3] for b in tiny_block_list)
    distributable_memory = total_memory - tiny_block_memory
    logger.debug(f"[MultiGPU_DisTorch2] Total Memory: {total_memory / (1024**2):.2f} MB, Tiny Block Memory: {tiny_block_memory / (1024**2):.2f} MB, Distributable Memory: {distributable_memory / (1024**2):.2f} MB")
    
    mode = "fraction"
    remaining_mem = 0 # Initialize for delayed logging
    if any(c in distorch_alloc.lower() for c in ['g', 'm', 'k', 'b']):
        mode = "byte"
    elif "%" in distorch_alloc:
        mode = "ratio"

    parsed_allocations = {}
    wildcard_device = "cpu" # Default

    raw_parsed = {}
    user_requested_values = {}
    for allocation in distorch_alloc.split(';'):
        if ',' not in allocation: continue
        dev_name, val_str = allocation.split(',', 1)
        if '*' in dev_name:
            dev_name = dev_name.replace('*','').strip()
            wildcard_device = dev_name
        
        try:
            if mode == "ratio":
                value = float(val_str.replace('%','').strip())
                raw_parsed[dev_name] = value
                user_requested_values[dev_name] = f"{value:.1f}%"
            elif mode == "byte":
                value_bytes = parse_memory_string(val_str)
                raw_parsed[dev_name] = value_bytes
                user_requested_values[dev_name] = f"{value_bytes / (1024**3):.2f}g"
            else: # fraction
                fraction = float(val_str)
                total_dev_mem = mm.get_total_memory(torch.device(dev_name))
                parsed_allocations[dev_name] = total_dev_mem * fraction
                user_requested_values[dev_name] = f"{int(fraction * 100)}%"
        except ValueError as e:
            logger.error(f"[MultiGPU_DisTorch2] Could not parse allocation '{allocation}': {e}")
            return

    if mode in ["ratio", "byte"]:
        total_requested = sum(raw_parsed.values())
        if mode == "ratio":
            for dev, val in raw_parsed.items():
                parsed_allocations[dev] = (val / total_requested) * distributable_memory
        elif mode == "byte":
            if total_requested > distributable_memory:
                logger.info(f"[MultiGPU_DisTorch2] Over-allocation: Requested {total_requested/(1024**3):.2f}GB, but model is {distributable_memory/(1024**3):.2f}GB. Pro-rating allocations.")
                for dev, val in raw_parsed.items():
                    parsed_allocations[dev] = (val / total_requested) * distributable_memory
            else:
                parsed_allocations = raw_parsed
                if wildcard_device not in parsed_allocations:
                    parsed_allocations[wildcard_device] = 0
                
                remaining_mem = distributable_memory - total_requested
                if remaining_mem > 0:
                    parsed_allocations[wildcard_device] += remaining_mem

    if wildcard_device not in parsed_allocations:
        parsed_allocations[wildcard_device] = 0

    # Sort devices for consistent processing
    sorted_devices = sorted(parsed_allocations.keys(), key=lambda d: (d == "cpu", d))

    if not distorch_alloc or distorch_alloc.isspace():
        logger.info("[MultiGPU_DisTorch2] Examples:")
        logger.info("  Direct(byte) Mode - cuda:0,500mb;cuda:1,3.0g;cpu,5gb* -> '*' cpu = over/underflow device, put 0.50gb on cuda0, 3.00gb on cuda1, and 5.00gb (or the rest) on cpu")
        logger.info("  Ratio(%) Mode - cuda:0,8%;cuda:1,8%;cpu,4% -> 8:8:4 ratio, put 40% on cuda0, 40% on cuda1, and 20% on cpu")
    else:
        if mode == "byte":
            feedback_parts = []
            for dev in sorted_devices:
                if dev in user_requested_values:
                    val_without_g = user_requested_values[dev].rstrip('g')
                    feedback_parts.append(f"{dev},{val_without_g}g")
                else:
                    feedback_parts.append(f"{dev},0.00g")
            
            wildcard_indicator = ""
            if wildcard_device:
                wildcard_indicator = f"*{wildcard_device}"
                
            logger.info(f"[MultiGPU_DisTorch2] Interpreted Byte Allocation: {';'.join(feedback_parts)}{wildcard_indicator}")
            
            original_parts = []
            original_wildcard_device = None
            for allocation in distorch_alloc.split(';'):
                if ',' not in allocation: continue
                dev_name, val_str = allocation.split(',', 1)
                if '*' in dev_name:
                    dev_name = dev_name.replace('*','').strip()
                    original_wildcard_device = dev_name
                original_parts.append((dev_name, val_str.strip()))
            
            if original_parts:
                formatted_parts = []
                for dev_name, val_str in original_parts:
                    if 'mb' in val_str.lower():
                        mb_val = float(val_str.lower().replace('mb', ''))
                        gb_val = mb_val / 1024
                        formatted_parts.append(f"{gb_val:.2f}gb on {dev_name}")
                    elif 'gb' in val_str.lower() or 'g' in val_str.lower():
                        val_num = float(''.join(filter(lambda x: x.isdigit() or x == '.', val_str)))
                        formatted_parts.append(f"{val_num:.2f}gb on {dev_name}")
                    else:
                        formatted_parts.append(f"{val_str} on {dev_name}")
                
                if formatted_parts:
                    if len(formatted_parts) == 1:
                        put_part = formatted_parts[0]
                    elif len(formatted_parts) == 2:
                        put_part = f"{formatted_parts[0]} and {formatted_parts[1]}"
                    else:
                        put_part = ", ".join(formatted_parts[:-1]) + f", and {formatted_parts[-1]}"
                    
                    wildcard_dev = original_wildcard_device if original_wildcard_device else "cpu"
                    logger.info(f"[MultiGPU_DisTorch2] Direct(byte) Mode - {distorch_alloc} -> '*' {wildcard_dev} = over/underflow device, put {put_part}")
            
            if remaining_mem > 0:
                logger.info(f"[MultiGPU_DisTorch2] Under-allocation: {remaining_mem/(1024**2):.2f}MB of model unallocated. Assigning to wildcard device '{wildcard_device}'.")

        elif mode == "ratio":
            total_requested_percent = sum(raw_parsed.values())
            if total_requested_percent > 0:
                normalized_ratios = {}
                for dev, percent in raw_parsed.items():
                    normalized_ratios[dev] = (percent / total_requested_percent) * 100
                
                ratio_parts = []
                ratio_values = []
                for allocation in distorch_alloc.split(';'):
                    if ',' not in allocation: continue
                    dev_name, val_str = allocation.split(',', 1)
                    val_str = val_str.strip()
                    if '%' in val_str:
                        ratio_val = val_str.replace('%', '').strip()
                        ratio_values.append(ratio_val)
                        ratio_parts.append(f"{dev_name}")
                
                if ratio_values and ratio_parts:
                    total_ratio = sum(float(val) for val in ratio_values)
                    normalized_pcts = []
                    for val in ratio_values:
                        normalized_pct = (float(val) / total_ratio) * 100
                        normalized_pcts.append(int(normalized_pct))
                    
                    ratio_string = ":".join(ratio_values)
                    
                    put_parts = []
                    for i, (dev_name, pct) in enumerate(zip(ratio_parts, normalized_pcts)):
                        put_parts.append(f"{pct}% on {dev_name}")
                    
                    if len(put_parts) == 1:
                        put_part = put_parts[0]
                    elif len(put_parts) == 2:
                        put_part = f"{put_parts[0]} and {put_parts[1]}"
                    else:
                        put_part = ", ".join(put_parts[:-1]) + f", and {put_parts[-1]}"
                    
                    logger.info(f"[MultiGPU_DisTorch2] Ratio(%) Mode - {distorch_alloc} -> {ratio_string} ratio, put {put_part}")

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Device Allocations")
    logger.info(eq_line)
    
    fmt_rosetta = "{:<10}{:>8}{:>8}{:>13}{:>10}"
    logger.info(fmt_rosetta.format("Device", "VRAM GB", "Dev %", "Model GB", "Dist Ratio"))
    logger.info(dash_line)
    
    dist_ratio_values = []
    if mode == "ratio":
        total_requested_percent = sum(raw_parsed.values())
        for dev in sorted_devices:
            if dev in raw_parsed:
                normalized_pct = (raw_parsed[dev] / total_requested_percent) * 100
                dist_ratio_values.append(f"{int(normalized_pct)}%")
            else:
                dist_ratio_values.append("0%")
    elif mode == "byte" or mode == "fraction":
        for dev in sorted_devices:
            model_percent = (parsed_allocations[dev] / distributable_memory) * 100 if distributable_memory > 0 else 0
            dist_ratio_values.append(f"{model_percent:.1f}%")

    for i, dev in enumerate(sorted_devices):
        alloc_gb = parsed_allocations[dev] / (1024**3)
        total_dev_gb = mm.get_total_memory(torch.device(dev)) / (1024**3)
        device_percent = (parsed_allocations[dev] / (total_dev_gb * 1024**3)) * 100 if total_dev_gb > 0 else 0
        
        dist_ratio_str = dist_ratio_values[i] if i < len(dist_ratio_values) else "N/A"
        
        logger.info(fmt_rosetta.format(dev, f"{total_dev_gb:.2f}", f"{device_percent:.1f}%", f"{alloc_gb:.2f}", dist_ratio_str))
    
    logger.info(dash_line)

    # Log layer distribution
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

    # Distribute blocks
    block_assignments = {}
    device_quotas = parsed_allocations.copy()
    
    compute_device = "cuda:0"
    for dev in sorted_devices:
        if dev != "cpu":
            compute_device = dev
            break

    devices_to_fill = sorted(device_quotas.keys(), key=lambda d: (d == "cpu", d))

    if mode == "ratio":
        total_requested_percent = sum(raw_parsed.values())
        normalized_ratios = {}
        if total_requested_percent > 0:
            for dev, percent in raw_parsed.items():
                normalized_ratios[dev] = (percent / total_requested_percent) * 100
        else:
            even_share = 100.0 / len(sorted_devices) if sorted_devices else 0
            for dev in sorted_devices:
                normalized_ratios[dev] = even_share
        
        exact_allocations = {}
        for dev in sorted_devices:
            if dev in normalized_ratios:
                exact_allocations[dev] = (normalized_ratios[dev] / 100) * total_memory
            else:
                exact_allocations[dev] = 0
        
        sorted_blocks = sorted(block_list, key=lambda b: b[3], reverse=True)
        
        device_remaining = exact_allocations.copy()
        for block_name, module, block_type, block_memory in sorted_blocks:
            best_device = None
            max_remaining = -1
            for device in sorted_devices:
                if device_remaining[device] >= block_memory and device_remaining[device] > max_remaining:
                    best_device = device
                    max_remaining = device_remaining[device]
            
            if best_device is None:
                best_device = compute_device
                
            block_assignments[block_name] = best_device
            device_remaining[best_device] -= block_memory
            
        unassigned_blocks = [b for b in block_list if b[0] not in block_assignments]
        if unassigned_blocks:
            unassigned_memory = sum(b[3] for b in unassigned_blocks)
            logger.warning(f"[MultiGPU_DisTorch2] {unassigned_memory / (1024**2):.2f} MB of model did not fit into ratio allocations. Assigning to compute device '{compute_device}'.")
            for block_name, _, _, _ in unassigned_blocks:
                block_assignments[block_name] = compute_device
    else:
        for block_name, module, block_type, block_memory in reversed(block_list):
            for device in devices_to_fill:
                if device_quotas.get(device, 0) >= block_memory:
                    block_assignments[block_name] = device
                    device_quotas[device] -= block_memory
                    break

        unassigned_blocks = [b for b in block_list if b[0] not in block_assignments]
        if unassigned_blocks:
            # This logic branch should not be hit if quotas are calculated correctly and a wildcard is used.
            # As a fallback, assign remaining blocks to the designated overflow device.
            if wildcard_device:
                unassigned_memory = sum(b[3] for b in unassigned_blocks)
                logger.info(f"[MultiGPU_DisTorch2] Assigning {len(unassigned_blocks)} remaining blocks ({unassigned_memory / (1024**2):.2f} MB) to overflow device '{wildcard_device}'.")
                for block_name, _, _, _ in unassigned_blocks:
                    block_assignments[block_name] = wildcard_device
            else:
                # If no wildcard is set, this is a true warning condition.
                unassigned_memory = sum(b[3] for b in unassigned_blocks)
                logger.warning(f"[MultiGPU_DisTorch2] {unassigned_memory / (1024**2):.2f} MB of model did not fit into allocations and no overflow device was set. Assigning to compute device '{compute_device}'.")
                for block_name, _, _, _ in unassigned_blocks:
                    block_assignments[block_name] = compute_device

    if mode == 'gb':
        total_non_wildcard_quota = sum(v for k, v in parsed_allocations.items() if k != wildcard_device)
        distributable_memory = sum(b[3] for b in block_list)
        if distributable_memory <= total_non_wildcard_quota:
            remaining_quota = total_non_wildcard_quota - distributable_memory
            logger.info(f"[MultiGPU_DisTorch2-GB] Underflow: Model fits in non-wildcard devices with {remaining_quota / (1024**2):.2f} MB to spare. Wildcard device '{wildcard_device}' will not be used for distributable blocks.")

    for block_name, _, _, _ in tiny_block_list:
        block_assignments[block_name] = compute_device

    # Populate final assignments for logging
    device_assignments = defaultdict(list)
    for block_name, device in block_assignments.items():
        for b_name, b_module, b_type, b_mem in all_blocks:
            if b_name == block_name:
                device_assignments[device].append((b_name, b_module, b_type, b_mem))
                break

    # Log final assignments
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

    device_memories = {}
    for device, blocks in device_assignments.items():
        dist_blocks = [b for b in blocks if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks: continue
        device_memories[device] = sum(b[3] for b in dist_blocks)

    final_sorted_devices = sorted(device_memories.keys(), key=lambda d: (d == "cpu", d))

    for dev in final_sorted_devices:
        dist_blocks = [b for b in device_assignments[dev] if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks: continue
        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_assign.format(dev, str(len(dist_blocks)), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))
    
    logger.info(dash_line)

    return {
        "device_assignments": device_assignments,
        "block_assignments": block_assignments
    }

def calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str):
    """Calculate virtual VRAM allocation string for distributed safetensor loading"""
    recipient_device, vram_amount, donors = virtual_vram_str.split(';')
    virtual_vram_gb = float(vram_amount)

    # EXACT SAME FORMATTING AS GGML
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
    
    allocation_string = ";".join(allocation_parts)
    
    fmt_mem = "{:<20}{:>20}"
    logger.info(fmt_mem.format("\n  v2 Expert String", allocation_string))

    return allocation_string

def override_class_with_distorch_safetensor_v2(cls):
    """DisTorch 2.0 wrapper for safetensor models"""
    from .nodes import get_device_list
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
