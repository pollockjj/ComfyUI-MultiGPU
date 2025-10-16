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
from .device_utils import get_device_list, soft_empty_cache_multigpu
from .model_management_mgpu import multigpu_memory_log, force_full_system_cleanup


def register_patched_safetensor_modelpatcher():
    """Register and patch the ModelPatcher for distributed safetensor loading"""
    from comfy.model_patcher import wipe_lowvram_weight, move_weight_functions
    # Patch ComfyUI's ModelPatcher
    if not hasattr(comfy.model_patcher.ModelPatcher, '_distorch_patched'):


        # PATCH load_models_gpu with correct memory calculations per model flags
        original_load_models_gpu = mm.load_models_gpu

        def patched_load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
            from comfy.model_management import cleanup_models_gc, get_free_memory, free_memory, current_loaded_models
            from comfy.model_management import VRAMState, vram_state, lowvram_available, MIN_WEIGHT_MEMORY_RATIO
            from comfy.model_management import minimum_inference_memory, extra_reserved_memory, is_device_cpu
            
            multigpu_memory_log("load_models_gpu_top_level", "start")

            cleanup_models_gc()

            inference_memory = minimum_inference_memory()
            extra_reserved_mem = extra_reserved_memory()
            memory_required_total = memory_required + extra_reserved_mem
            extra_mem = max(inference_memory, memory_required_total)
            if minimum_memory_required is None:
                minimum_memory_required = extra_mem
            else:
                minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_mem)

            models_temp = set()
            for m in models:
                models_temp.add(m)
                model_type = type(m).__name__

                if ("GGUF" in model_type or "ModelPatcher" in model_type) and hasattr(m, "model_patches_to"):
                    logger.info(f"[MultiGPU DisTorch V2] {type(m).__name__} missing 'model_patches_models' attribute, using 'model_patches_to' fallback.")
                    target_device = m.load_device
                    logger.debug(f"[MultiGPU DisTorch V2] Target device: {target_device}")
                    patches = m.model_patches_to(target_device)
                    if patches:
                        logger.debug(f"[MultiGPU DisTorch V2] Found {len(patches)} mm_patch(es) for {type(m).__name__} on device {target_device}")
                        for mm_patch in patches:
                            logger.debug(f"[MultiGPU DisTorch V2] Registering mm_patch: {type(mm_patch).__name__}")
                            models_temp.add(mm_patch)
                    continue
                
                for mm_patch in m.model_patches_models():
                    models_temp.add(mm_patch)
                patches = m.model_patches_to(m.load_device)
                if patches:
                    for mm_patch in patches:
                        models_temp.add(mm_patch)

            models = models_temp

            models_to_load = []

            for x in models:
                loaded_model = mm.LoadedModel(x)
                try:
                    loaded_model_index = current_loaded_models.index(loaded_model)
                except:
                    loaded_model_index = None

                if loaded_model_index is not None:
                    loaded = current_loaded_models[loaded_model_index]
                    loaded.currently_used = True
                    models_to_load.append(loaded)
                else:
                    if hasattr(x, "model"):
                        logging.info(f"Requested to load {x.model.__class__.__name__}")
                    models_to_load.append(loaded_model)

            for loaded_model in models_to_load:
                to_unload = []
                for i in range(len(current_loaded_models)):
                    if loaded_model.model.is_clone(current_loaded_models[i].model):
                        to_unload = [i] + to_unload
                for i in to_unload:
                    model_to_unload = current_loaded_models.pop(i)
                    model_to_unload.model.detach(unpatch_all=False)
                    model_to_unload.model_finalizer.detach()

            # DisTorch Processing
            total_memory_required = {}
            eject_device = None

            for loaded_model in models_to_load:
                device = loaded_model.device
                base_memory = loaded_model.model_memory_required(device)

                inner_model = loaded_model.model.model
                
                if hasattr(inner_model, '_distorch_v2_meta'):
                    meta = inner_model._distorch_v2_meta
                    allocation_str = meta['full_allocation']
                    
                    # Parse allocation string: "expert#compute_device;virtual_vram_gb;donors"
                    parts = allocation_str.split('#')
                    virtual_vram_gb = 0.0
                    has_eject = False
                    
                    if len(parts) > 1:
                        virtual_vram_str = parts[1]
                        virtual_info = virtual_vram_str.split(';')
                        if len(virtual_info) > 1:
                            virtual_vram_gb = float(virtual_info[1])
                        if len(virtual_info) > 2 and virtual_info[2]:
                            has_eject = True
                    
                    if has_eject:
                        eject_device = device
                        logger.mgpu_mm_log("DisTorch eject_models detected - MAX memory eviction")
                    
                    virtual_vram_bytes = virtual_vram_gb * (1024**3)
                    adjusted_memory = max(0, base_memory - virtual_vram_bytes)
                    total_memory_required[device] = total_memory_required.get(device, 0) + adjusted_memory
                    logger.mgpu_mm_log(f"DisTorch model adjusted {(base_memory - virtual_vram_bytes)/(1024**3):.2f}GB for device {device}")
                else:
                    # Standard model: use full model size
                    total_memory_required[device] = total_memory_required.get(device, 0) + base_memory
                    logger.mgpu_mm_log(f"[LOAD_MODELS_GPU] Standard model {(base_memory)/(1024**3):.2f}GB for device {device}")

            for device in total_memory_required:
                if device != torch.device("cpu"):
                    requested_mem = total_memory_required[device] * 1.1 + extra_mem
                    logger.mgpu_mm_log(f"[FREE_MEMORY_CALL] Device {device}: requesting {requested_mem/(1024**3):.2f}GB = {total_memory_required[device]/(1024**3):.2f}GB * 1.1 + {extra_mem/(1024**3):.2f}GB inference")
            
            
            multigpu_memory_log("free_memory", "pre")

            for device in total_memory_required:
                if device != torch.device("cpu"):
                    if device == eject_device:
                        total_device_memory = mm.get_total_memory(device)
                        logger.mgpu_mm_log(f"[LOAD_MODELS_GPU] eject_models=1, is_distorch=1 → using MAX memory ({total_device_memory/(1024**3):.2f}GB) for eviction")
                        free_memory(total_device_memory,device)
                    else:
                        logger.mgpu_mm_log(f"[LOAD_MODELS_GPU] eject_models=0, using Comfy Core Computed memory ({(total_memory_required[device] * 1.1 + extra_mem)/(1024**3):.2f}GB) for eviction")
                        free_memory(total_memory_required[device] * 1.1 + extra_mem, device)
            
            multigpu_memory_log("free_memory/minimum_memory_required", "post/pre")

            for device in total_memory_required:
                if device != torch.device("cpu"):
                    free_mem = get_free_memory(device)
                    free_mem_gb = free_mem / (1024**3)
                    min_required_gb = minimum_memory_required / (1024**3)
                    logger.mgpu_mm_log(f"[MIN_MEMORY_CHECK] Device {device}: free={free_mem_gb:.2f}GB, required={min_required_gb:.2f}GB, will_evict={free_mem < minimum_memory_required}")

                    if free_mem < minimum_memory_required:
                        models_l = free_memory(minimum_memory_required, device)
                        logger.mgpu_mm_log(f"[EVICTION] Device {device}: unloaded {len(models_l)} models due to insufficient memory")
                        logging.info("{} models unloaded.".format(len(models_l)))

            multigpu_memory_log("minimum_memory_required", "post")

            for loaded_model in models_to_load:
                model = loaded_model.model
                torch_dev = model.load_device
                if is_device_cpu(torch_dev):
                    vram_set_state = VRAMState.DISABLED
                else:
                    vram_set_state = vram_state
                lowvram_model_memory = 0
                if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM) and not force_full_load:
                    loaded_memory = loaded_model.model_loaded_memory()
                    current_free_mem = get_free_memory(torch_dev) + loaded_memory

                    lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
                    lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

                if vram_set_state == VRAMState.NO_VRAM:
                    lowvram_model_memory = 0.1

                loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
                current_loaded_models.insert(0, loaded_model)

        # Replace the module function
        mm.load_models_gpu = patched_load_models_gpu

        original_partially_load = comfy.model_patcher.ModelPatcher.partially_load

        def new_partially_load(self, device_to, extra_memory=0, full_load=False, force_patch_weights=False, **kwargs):
            """Override to use direct model annotation for allocation"""
            
            mp_id = id(self)
            mp_patches_uuid = self.patches_uuid
            inner_model = self.model
            inner_model_id = id(inner_model)
            
            if not hasattr(inner_model, "_distorch_v2_meta"):
                logger.debug(f"[DISTORCH_SKIP] ModelPatcher=0x{mp_id:x} inner_model=0x{inner_model_id:x} type={type(inner_model).__name__} - no metadata, using standard loading")
                result = original_partially_load(self, device_to, extra_memory, force_patch_weights)
                if hasattr(self, '_distorch_block_assignments'):
                    del self._distorch_block_assignments
                return result
            
            allocations = inner_model._distorch_v2_meta['full_allocation']
            
            if not hasattr(self.model, '_distorch_high_precision_loras'):
                self.model._distorch_high_precision_loras = True

            if not hasattr(self.model, 'current_weight_patches_uuid'):
                self.model.current_weight_patches_uuid = None

            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)

            if unpatch_weights:
                logger.debug(f"[MultiGPU DisTorch V2] Patches changed or forced. Unpatching model.")
                self.unpatch_model(self.offload_device, unpatch_weights=True)

            self.patch_model(load_weights=False)

            mem_counter = 0

            is_clip_model = getattr(self, 'is_clip', False)
            device_assignments = analyze_safetensor_loading(self, allocations, is_clip=is_clip_model)
            
            model_original_dtype = comfy.utils.weight_dtype(self.model.state_dict())
            high_precision_loras = getattr(self.model, "_distorch_high_precision_loras", True)
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
                        logger.debug(f"[MultiGPU DisTorch V2] Moving already patched {module_name} to {block_target_device}")
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
                            logger.debug(f"[MultiGPU DisTorch V2] Cast {module_name}.{param_name} to FP8 for CPU storage")

                # Step 4: Move to ultimate destination based on DisTorch assignment
                if block_target_device != device_to:
                    logger.debug(f"[MultiGPU DisTorch V2] Moving {module_name} from {device_to} to {block_target_device}")
                    module_object.to(block_target_device)
                    module_object.comfy_cast_weights = True

                # Mark as patched and update memory counter
                module_object.comfy_patched_weights = True
                mem_counter += module_size

            self.model.current_weight_patches_uuid = self.patches_uuid

            self.model.device = device_to

            logger.info("[MultiGPU DisTorch V2] DisTorch loading completed.")
            logger.info(f"[MultiGPU DisTorch V2] Total memory: {mem_counter / (1024 * 1024):.2f}MB")

            return 0

        
        comfy.model_patcher.ModelPatcher.partially_load = new_partially_load
        comfy.model_patcher.ModelPatcher._distorch_patched = True
        logger.info("[MultiGPU Core Patching] Successfully patched ModelPatcher.partially_load")

def _extract_clip_head_blocks(raw_block_list, compute_device):
    """Identify and pre-assign CLIP head blocks to compute device returning head_blocks, distributable_blocks, block_assignments, and head_memory."""
    head_keywords = ['embed', 'wte', 'wpe', 'token_embedding', 'position_embedding']
    head_blocks = []
    distributable_blocks = []
    head_memory = 0
    block_assignments = {}
    
    for module_size, module_name, module_object, params in raw_block_list:
        if any(kw in module_name.lower() for kw in head_keywords):
            head_blocks.append((module_size, module_name, module_object, params))
            block_assignments[module_name] = compute_device
            head_memory += module_size
        else:
            distributable_blocks.append((module_size, module_name, module_object, params))
    
    return head_blocks, distributable_blocks, block_assignments, head_memory

def analyze_safetensor_loading(model_patcher, allocations_string, is_clip=False):
    """
    Analyze and distribute safetensor model blocks across devices.
    Supports CLIP head preservation when is_clip=True.
    """
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = ""
    virtual_vram_str = ""
    virtual_vram_gb = 0.0

    if '#' in allocations_string:
        distorch_alloc, virtual_vram_str = allocations_string.split('#', 1)
    else:
        distorch_alloc = allocations_string

    compute_device = virtual_vram_str.split(';')[0] if virtual_vram_str else "cuda:0"
    logger.debug(f"[MultiGPU DisTorch V2] Compute Device: {compute_device}")

    if not distorch_alloc:
        mode = "fraction"
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

    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    logger.info(eq_line)
    logger.info(f"[MultiGPU DisTorch V2] Final Allocation String:\n{distorch_alloc}")

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
    logger.debug(f"[MultiGPU DisTorch V2] Total model memory: {total_memory} bytes")
    logger.debug(f"[MultiGPU DisTorch V2] Tiny block threshold (0.01%): {MIN_BLOCK_THRESHOLD} bytes")

    # CLIP-specific: Extract head blocks and get pre-assignments
    head_memory = 0
    block_assignments = {}
    if is_clip:
        head_blocks, distributable_raw, block_assignments, head_memory = \
            _extract_clip_head_blocks(raw_block_list, compute_device)
        logger.info(f"[MultiGPU DisTorch V2 CLIP] Preserving {len(head_blocks)} head layer(s) ({head_memory/(1024**2):.2f} MB) on compute device: {compute_device}")
    else:
        distributable_raw = raw_block_list

    # Build all_blocks list for summary (using full raw_block_list)
    all_blocks = []
    for module_size, module_name, module_object, params in raw_block_list:
        block_type = type(module_object).__name__
        # Populate summary dictionaries
        block_summary[block_type] = block_summary.get(block_type, 0) + 1
        memory_by_type[block_type] += module_size
        all_blocks.append((module_name, module_object, block_type, module_size))

    # Use distributable blocks for actual allocation (for CLIP, this excludes heads)
    distributable_all_blocks = []
    for module_size, module_name, module_object, params in distributable_raw:
        distributable_all_blocks.append((module_name, module_object, type(module_object).__name__, module_size))

    block_list = [b for b in distributable_all_blocks if b[3] >= MIN_BLOCK_THRESHOLD]
    tiny_block_list = [b for b in distributable_all_blocks if b[3] < MIN_BLOCK_THRESHOLD]
    
    logger.debug(f"[MultiGPU DisTorch V2] Total blocks: {len(all_blocks)}")
    logger.debug(f"[MultiGPU DisTorch V2] Distributable blocks: {len(block_list)}")
    logger.debug(f"[MultiGPU DisTorch V2] Tiny blocks (<0.01%): {len(tiny_block_list)}")

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

    # CLIP-specific: Adjust compute_device quota to account for locked head blocks
    if is_clip and compute_device in donor_quotas and head_memory > 0:
        donor_quotas[compute_device] = max(0, donor_quotas[compute_device] - head_memory)
        logger.debug(f"[MultiGPU DisTorch V2 CLIP] Adjusted {compute_device} quota by -{head_memory/(1024**2):.2f} MB for head preservation")

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
        logger.debug(f"[MultiGPU DisTorch V2] Tiny block memory breakdown: {tiny_block_memory} bytes ({tiny_mem_mb:.2f} MB), which is {tiny_mem_percent:.4f}% of total model memory.")

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
    """Convert byte allocation string (e.g. 'cuda:1,4gb;cpu,*') to fractional VRAM allocation string respecting device order and byte quotas."""
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
            logger.info(f"[MultiGPU DisTorch V2] Assigning {bytes_to_assign / (1024**2):.2f}MB of model to {dev} (requested {requested_bytes / (1024**2):.2f}MB).")
        
        if remaining_model_bytes <= 0:
            logger.info("[MultiGPU DisTorch V2] All model blocks have been allocated. Subsequent devices in the string will receive no assignment.")
            break

    # Assign any leftover model bytes to the wildcard device
    if remaining_model_bytes > 0:
        final_byte_allocations[wildcard_device] += remaining_model_bytes
        logger.info(f"[MultiGPU DisTorch V2] Assigning remaining {remaining_model_bytes / (1024**2):.2f}MB of model to wildcard device '{wildcard_device}'.")

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
    """Convert ratio allocation string (e.g. 'cuda:0,25%;cpu,75%') describing model split to fractional VRAM allocation string."""
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
    
    logger.info(f"[MultiGPU DisTorch V2] Ratio(%) Mode - {ratio_str} -> {ratio_string} ratio, put {put_part}")

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
        logger.warning(f"\n\n[MultiGPU DisTorch V2] Model size ({model_size_gb:.2f}GB) is larger than 90% of available VRAM on: {recipient_device} ({recipient_vram * 0.9:.2f}GB).")
        logger.warning(f"[MultiGPU DisTorch V2] To prevent an OOM error, set 'virtual_vram_gb' to at least {required_offload_gb:.2f}.\n\n")

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
