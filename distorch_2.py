"""
DisTorch Safetensor Memory Management Module
Contains all safetensor related code for distributed memory management
Following the ethos: leverage ComfyUI core, monkey-patch minimally, don't rewrite
"""

import torch
import logging
import hashlib
import copy
from collections import defaultdict
import comfy.model_management as mm
import comfy.model_patcher

logger = logging.getLogger("MultiGPU")

# Global store for safetensor model allocations
safetensor_allocation_store = {}
safetensor_settings_store = {}


def create_safetensor_model_hash(model, caller):
    """Create a unique hash for a safetensor model to track allocations - EXACTLY like GGUF"""
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
    logger.debug(f"[MULTIGPU_DISTORCHV2_HASH] Created hash for {caller}: {final_hash[:8]}...")
    return final_hash


def register_patched_safetensor_modelpatcher():
    """Register the PROPERLY IMPLEMENTED monkey-patch for ModelPatcher"""
    
    if not hasattr(comfy.model_patcher.ModelPatcher, '_distorch_patched'):
        # Store original methods
        original_partially_load = comfy.model_patcher.ModelPatcher.partially_load
        original_load = comfy.model_patcher.ModelPatcher.load
        
        def new_partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            """
            Enhanced DisTorch2 partially_load that sets up block assignments
            """
            global safetensor_allocation_store
            
            # Check if we have allocations for this model
            model_hash = create_safetensor_model_hash(self, "partial_load")
            allocations = safetensor_allocation_store.get(model_hash)
            
            if allocations:
                logger.info(f"[DISTORCH2] Setting up device distribution for model {model_hash[:8]}")
                
                # Parse allocations to get device assignments
                device_assignments = analyze_safetensor_loading(self, allocations)
                block_assignments = device_assignments['block_assignments']
                
                # Count CPU vs GPU assignments for logging
                cpu_blocks = sum(1 for d in block_assignments.values() if d == "cpu")
                gpu_blocks = len(block_assignments) - cpu_blocks
                logger.info(f"[DISTORCH2] Block distribution: {gpu_blocks} GPU, {cpu_blocks} CPU")
                
                # Store block assignments on the object
                self._distorch_block_assignments = block_assignments
            
            # Call original
            result = original_partially_load(self, device_to, extra_memory, force_patch_weights)
            
            # Clean up
            if hasattr(self, '_distorch_block_assignments'):
                del self._distorch_block_assignments
                
            return result
        
        def new_load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
            """
            Modified load that respects DisTorch lowvram decisions
            This is where we intercept ComfyUI's lowvram decision AND handle GPU-to-GPU transfers
            """
            # Check if we have DisTorch assignments
            has_distorch = hasattr(self, '_distorch_block_assignments')
            
            if has_distorch:
                block_assignments = self._distorch_block_assignments
                logger.info(f"[DISTORCH2] Intercepting load with custom block assignments")
                
                # DEBUG: Log what we're about to do
                cpu_blocks = [k for k, v in block_assignments.items() if v == "cpu"]
                gpu_blocks = {dev: [k for k, v in block_assignments.items() if v == dev and dev != "cpu"] 
                             for dev in set(block_assignments.values()) if dev != "cpu"}
                
                logger.info(f"[DISTORCH2 DEBUG] CPU blocks to offload: {len(cpu_blocks)}")
                for gpu_dev, blocks in gpu_blocks.items():
                    logger.info(f"[DISTORCH2 DEBUG] {gpu_dev} blocks to assign: {len(blocks)}")
                
                # CRITICAL FIX: Force full_load=False when we have DisTorch assignments
                if full_load:
                    logger.warning(f"[DISTORCH2] Overriding full_load=True to False for distributed loading")
                    full_load = False
            
            # Mostly copy ComfyUI's load logic but with our intercept
            with self.use_ejected():
                self.unpatch_hooks()
                mem_counter = 0
                patch_counter = 0
                lowvram_counter = 0
                loading = self._load_list()

                # Track which modules go to which device for GPU-to-GPU transfers
                gpu_device_moves = {} if has_distorch else None

                load_completely = []
                loading.sort(reverse=True)
                for x in loading:
                    n = x[1]
                    m = x[2]
                    params = x[3]
                    module_mem = x[0]

                    lowvram_weight = False

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)

                    # DISTORCH INTERCEPT: Handle both CPU and GPU assignments
                    if has_distorch and n in block_assignments:
                        target_device = block_assignments[n]
                        if target_device == "cpu":
                            # CPU offload through lowvram mechanism
                            lowvram_weight = True
                            logger.info(f"[DISTORCH2 DEBUG] Block {n} marked for CPU offload (lowvram)")
                        elif target_device != device_to:
                            # GPU-to-GPU transfer - track for later movement
                            if gpu_device_moves is not None:
                                gpu_device_moves[n] = (m, target_device)
                            logger.debug(f"[DISTORCH2] Marking {n} for {target_device} (GPU transfer)")
                    elif not full_load and hasattr(m, "comfy_cast_weights"):
                        # Original ComfyUI logic (only if we don't have a DisTorch assignment)
                        if mem_counter + module_mem >= lowvram_model_memory:
                            lowvram_weight = True
                            lowvram_counter += 1
                            if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                                logger.debug(f"[DISTORCH2 DEBUG] Skipping {n} - already lowvramed (prev_comfy_cast_weights exists)")
                                continue

                    cast_weight = self.force_cast_weights
                    if lowvram_weight:
                        if hasattr(m, "comfy_cast_weights"):
                            m.weight_function = []
                            m.bias_function = []

                        # DEFER TO COMFYUI'S LOGIC TO CREATE LowVramPatch FOR ALL PARAMETERS
                        for param_name in params:
                            param_key = f"{n}.{param_name}"
                            if param_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(param_key)
                                else:
                                    from comfy.model_patcher import LowVramPatch
                                    # Dynamically assign to weight_function or bias_function
                                    if 'bias' in param_name:
                                        if not hasattr(m, 'bias_function'):
                                            m.bias_function = []
                                        m.bias_function.append(LowVramPatch(param_key, self.patches))
                                    else:
                                        if not hasattr(m, 'weight_function'):
                                            m.weight_function = []
                                        m.weight_function.append(LowVramPatch(param_key, self.patches))
                                    patch_counter += 1
                                    logger.info(f"[DISTORCH2] Created LowVramPatch for {param_key}")
                        
                        # This is a proxy for ComfyUI's internal counter
                        if any(f"{n}.{p}" in self.patches for p in params):
                            lowvram_counter += 1

                        cast_weight = True
                    else:
                        if hasattr(m, "comfy_cast_weights"):
                            from comfy.model_patcher import wipe_lowvram_weight
                            wipe_lowvram_weight(m)

                        if full_load or mem_counter + module_mem < lowvram_model_memory:
                            mem_counter += module_mem
                            load_completely.append((module_mem, n, m, params))

                    if cast_weight and hasattr(m, "comfy_cast_weights"):
                        m.prev_comfy_cast_weights = m.comfy_cast_weights
                        m.comfy_cast_weights = True

                    if weight_key in self.weight_wrapper_patches:
                        m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                    if bias_key in self.weight_wrapper_patches:
                        m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                    from comfy.model_patcher import move_weight_functions
                    mem_counter += move_weight_functions(m, device_to)

                load_completely.sort(reverse=True)
                for x in load_completely:
                    n = x[1]
                    m = x[2]
                    params = x[3]
                    if hasattr(m, "comfy_patched_weights"):
                        if m.comfy_patched_weights == True:
                            continue

                    for param in params:
                        self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                    logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                    m.comfy_patched_weights = True

                for x in load_completely:
                    x[2].to(device_to)

                # CRITICAL: Handle GPU-to-GPU transfers AFTER normal loading
                if gpu_device_moves:
                    logger.info(f"[DISTORCH2] Performing GPU-to-GPU transfers for {len(gpu_device_moves)} modules")
                    for module_name, (module, target_device) in gpu_device_moves.items():
                        try:
                            # Move the module to its target GPU
                            module.to(target_device)
                            logger.debug(f"[DISTORCH2] Moved {module_name} to {target_device}")
                            
                            # Update memory tracking
                            module_size = comfy.model_management.module_size(module)
                            # Note: This is approximate - we're removing from primary device memory
                            # and adding to target device, but ComfyUI doesn't fully track multi-GPU
                            mem_counter -= module_size
                            
                        except Exception as e:
                            logger.error(f"[DISTORCH2] Failed to move {module_name} to {target_device}: {e}")

                if lowvram_counter > 0:
                    logger.info(f"[DISTORCH2 SUCCESS] Created {lowvram_counter} lowvram patches, {patch_counter} total patches")
                    logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                    self.model.model_lowvram = True
                else:
                    # DEBUG: Log why we're loading completely
                    logger.info(f"[DISTORCH2 DEBUG] Loading completely - full_load={full_load}, lowvram_counter={lowvram_counter}")
                    logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                    self.model.model_lowvram = False
                    if full_load:
                        self.model.to(device_to)
                        mem_counter = self.model_size()

                self.model.lowvram_patch_counter += patch_counter
                self.model.device = device_to
                self.model.model_loaded_weight_memory = mem_counter
                self.model.current_weight_patches_uuid = self.patches_uuid

                from comfy.patcher_extension import CallbacksMP
                for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                    callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

                self.apply_hooks(self.forced_hooks, force_apply=True)
                
                # Final debug log with verification
                if has_distorch:
                    # Verify LowVramPatch creation for CPU blocks
                    cpu_blocks_with_patches = 0
                    cpu_blocks_verified = []
                    for n in block_assignments:
                        if block_assignments[n] == "cpu":
                            # Find the module
                            for x in loading:
                                if x[1] == n:
                                    m = x[2]
                                    has_weight_func = hasattr(m, 'weight_function') and len(m.weight_function) > 0
                                    has_bias_func = hasattr(m, 'bias_function') and len(m.bias_function) > 0
                                    if has_weight_func or has_bias_func:
                                        cpu_blocks_with_patches += 1
                                        cpu_blocks_verified.append(n)
                                    else:
                                        logger.warning(f"[DISTORCH2 WARNING] CPU block {n} has NO LowVramPatch!")
                                    break
                    
                    logger.info(f"[DISTORCH2 FINAL] Load complete - lowvram patches created: {lowvram_counter}, patch_counter: {patch_counter}, GPU transfers: {len(gpu_device_moves) if gpu_device_moves else 0}")
                    logger.info(f"[DISTORCH2 VERIFY] CPU blocks with patches: {cpu_blocks_with_patches}/{len([b for b in block_assignments.values() if b == 'cpu'])}")
                    if cpu_blocks_with_patches < 10 and len(cpu_blocks_verified) > 0:
                        logger.info(f"[DISTORCH2 VERIFY] Verified CPU blocks: {cpu_blocks_verified[:10]}")
        
        # Apply the monkey-patches
        comfy.model_patcher.ModelPatcher.partially_load = new_partially_load
        comfy.model_patcher.ModelPatcher.load = new_load
        comfy.model_patcher.ModelPatcher._distorch_patched = True
        logger.info("[DISTORCH2] Successfully patched ModelPatcher to intercept lowvram decisions")


def analyze_safetensor_loading(model_patcher, allocations_str):
    """
    Analyze and distribute safetensor model blocks across devices
    IDENTICAL LOGGING FORMAT TO analyze_ggml_loading
    """
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = allocations_str
    virtual_vram_gb = 0.0

    # Parse allocation string EXACTLY like GGML
    if '#' in allocations_str:
        distorch_alloc, virtual_vram_str = allocations_str.split('#')
        if not distorch_alloc:
            distorch_alloc = calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str)

    # EXACT SAME FORMATTING AS GGML
    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    # Parse device allocations
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

    # IDENTICAL LOGGING TO DISTORCH
    logger.info(eq_line)
    logger.info("    DisTorch2 Model Device Allocations")
    logger.info(eq_line)
    logger.info(fmt_assign.format("Device", "Alloc %", "Total (GB)", " Alloc (GB)"))
    logger.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_devices:
        frac = device_table[dev]["fraction"]
        tot_gb = device_table[dev]["total_gb"]
        alloc_gb = device_table[dev]["alloc_gb"]
        logger.info(fmt_assign.format(dev,f"{int(frac * 100)}%",f"{tot_gb:.2f}",f"{alloc_gb:.2f}"))

    logger.info(dash_line)

    # Analyze model blocks using ComfyUI's structure
    block_summary = {}
    block_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    # Get the actual model from the patcher
    model = model_patcher.model if hasattr(model_patcher, 'model') else model_patcher

    # First pass: calculate total memory to establish threshold
    total_memory = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") or hasattr(module, "comfy_cast_weights"):
            try:
                block_memory = mm.module_size(module)
            except:
                block_memory = 0
                if hasattr(module, 'weight') and module.weight is not None:
                    block_memory += module.weight.numel() * module.weight.element_size()
                if hasattr(module, 'bias') and module.bias is not None:
                    block_memory += module.bias.numel() * module.bias.element_size()
            total_memory += block_memory

    # Set the minimum block size threshold (0.01% of total model memory)
    MIN_BLOCK_THRESHOLD = total_memory * 0.0001
    logger.debug(f"[MultiGPU_DisTorch2] Total model memory: {total_memory} bytes")
    logger.debug(f"[MultiGPU_DisTorch2] Tiny block threshold (0.01%): {MIN_BLOCK_THRESHOLD} bytes")

    # Second pass: analyze and collect all blocks, then filter
    all_blocks = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") or hasattr(module, "comfy_cast_weights"):
            block_type = type(module).__name__
            
            try:
                block_memory = mm.module_size(module)
            except:
                block_memory = 0
                if hasattr(module, 'weight') and module.weight is not None:
                    block_memory += module.weight.numel() * module.weight.element_size()
                if hasattr(module, 'bias') and module.bias is not None:
                    block_memory += module.bias.numel() * module.bias.element_size()
            
            # Populate summary dictionaries with ALL blocks for accurate reporting
            block_summary[block_type] = block_summary.get(block_type, 0) + 1
            memory_by_type[block_type] += block_memory
            all_blocks.append((name, module, block_type, block_memory))

    # Filter out tiny blocks from the distribution list
    block_list = [b for b in all_blocks if b[3] >= MIN_BLOCK_THRESHOLD]
    tiny_block_list = [b for b in all_blocks if b[3] < MIN_BLOCK_THRESHOLD]
    
    logger.debug(f"[MultiGPU_DisTorch2] Total blocks: {len(all_blocks)}")
    logger.debug(f"[MultiGPU_DisTorch2] Distributable blocks: {len(block_list)}")
    logger.debug(f"[MultiGPU_DisTorch2] Tiny blocks (<0.01%): {len(tiny_block_list)}")

    # Log layer distribution - IDENTICAL FORMAT TO GGML
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

    # Determine the primary compute device (first non-cpu device)
    compute_device = "cuda:0" # Fallback
    for dev in sorted_devices:
        if dev != "cpu":
            compute_device = dev
            break
            
    # Calculate total memory to be offloaded to donor devices
    total_offload_gb = sum(DEVICE_RATIOS_DISTORCH.get(d, 0) for d in sorted_devices if d != compute_device)
    total_offload_bytes = total_offload_gb * (1024**3)
    
    offloaded_bytes = 0
    
    # Iterate from the TAIL of the model
    for block_name, module, block_type, block_memory in reversed(block_list):
        try:
            # block_memory is already calculated
            pass
        except:
            block_memory = 0
            if hasattr(module, 'weight') and module.weight is not None:
                block_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, 'bias') and module.bias is not None:
                block_memory += module.bias.numel() * module.bias.element_size()

        # Assign to donor device (currently assumes one donor 'cpu') until target is met
        if offloaded_bytes < total_offload_bytes:
            # For now, simple offload to CPU, will expand for multi-donor
            donor_device = "cpu"
            for dev in sorted_devices:
                if dev != compute_device:
                    donor_device = dev
                    break # Use first available donor
            
            block_assignments[block_name] = donor_device
            offloaded_bytes += block_memory
        else:
            # Assign remaining blocks to the primary compute device
            block_assignments[block_name] = compute_device

    # Explicitly assign tiny blocks to the compute device
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

    # Log final assignments - IDENTICAL FORMAT TO GGML
    logger.info("DisTorch2 Model Final Device/Layer Assignments")
    logger.info(dash_line)
    logger.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    
    # Calculate and log tiny blocks separately
    if tiny_block_list:
        tiny_block_memory = sum(b[3] for b in tiny_block_list)
        tiny_mem_mb = tiny_block_memory / (1024 * 1024)
        tiny_mem_percent = (tiny_block_memory / total_memory) * 100 if total_memory > 0 else 0
        device_label = f"{compute_device} (<0.01%)"
        logger.info(fmt_assign.format(device_label, str(len(tiny_block_list)), f"{tiny_mem_mb:.2f}", f"{tiny_mem_percent:.1f}%"))
        logger.debug(f"[MultiGPU_DisTorch2] Tiny block memory breakdown: {tiny_block_memory} bytes ({tiny_mem_mb:.2f} MB), which is {tiny_mem_percent:.4f}% of total model memory.")

    # Log distributed blocks
    total_assigned_memory = 0
    device_memories = {}
    
    for device, blocks in device_assignments.items():
        # Exclude tiny blocks from this calculation
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
    ram_donors = [d for d in donors.split(',') if d != 'cpu']
    remaining_vram_needed = virtual_vram_gb
    
    donor_device_info = {}
    donor_allocations = {}
    
    for donor in ram_donors:
        donor_vram = mm.get_total_memory(torch.device(donor)) / (1024**3)
        max_donor_capacity = donor_vram * 0.9  # Use 90% max
        
        donation = min(remaining_vram_needed, max_donor_capacity)
        donor_virtual = donor_vram - donation
        remaining_vram_needed -= donation
        donor_allocations[donor] = donation
            
        donor_device_info[donor] = (donor_vram, donor_virtual)
        logger.info(fmt_assign.format(donor, 'donor', f"{donor_vram:.2f}GB",  f"{donor_virtual:.2f}GB", f"-{donation:.2f}GB"))
    
    # CPU gets the rest
    system_dram_gb = mm.get_total_memory(torch.device('cpu')) / (1024**3)
    cpu_donation = remaining_vram_needed
    cpu_virtual = system_dram_gb - cpu_donation
    donor_allocations['cpu'] = cpu_donation
    logger.info(fmt_assign.format('cpu', 'donor', f"{system_dram_gb:.2f}GB", f"{cpu_virtual:.2f}GB", f"-{cpu_donation:.2f}GB"))
    
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
    
    cpu_percent = donor_allocations['cpu'] / system_dram_gb
    allocation_parts.append(f"cpu,{cpu_percent:.4f}")

    allocation_string = ";".join(allocation_parts)
    
    fmt_mem = "{:<20}{:>20}"
    logger.info(fmt_mem.format("\n  v2 Expert String", allocation_string))

    return allocation_string


def override_class_with_distorch_safetensor_v2(cls):
    """DisTorch 2.0 wrapper for safetensor models - EXACTLY like GGUF wrapper"""
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
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch2)"

        @classmethod
        def IS_CHANGED(s, *args, compute_device=None, virtual_vram_gb=4.0, 
                       donor_device="cpu", expert_mode_allocations="", **kwargs):
            # Create a hash of our specific settings
            settings_str = f"{compute_device}{virtual_vram_gb}{donor_device}{expert_mode_allocations}"
            return hashlib.sha256(settings_str.encode()).hexdigest()

        def override(self, *args, compute_device=None, virtual_vram_gb=4.0, 
                     donor_device="cpu", expert_mode_allocations="", **kwargs):
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
                    # The IS_CHANGED mechanism should handle the reload, this is for logging.
                else:
                    logger.info(f"[MultiGPU_DisTorch2] Settings unchanged for model {model_hash[:8]}. Using cached model.")

            out = fn(*args, **kwargs)

            # Build allocation string - EXACTLY like GGUF
            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{compute_device};{virtual_vram_gb};{donor_device}"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            logger.info(f"[MULTIGPU_DISTORCHV2] Full allocation string: {full_allocation}")
            
            # Store allocation for the model - EXACTLY like GGUF
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
