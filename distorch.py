"""
DisTorch GGUF/GGML Memory Management Module
Contains all GGUF/GGML related code for distributed memory management
"""

import sys
import torch
import logging
import hashlib

logger = logging.getLogger("MultiGPU")
import copy
from collections import defaultdict
import comfy.model_management as mm

# Global store for model allocations
model_allocation_store = {}


def create_model_hash(model, caller):
    """Create a unique hash for a model to track allocations"""
    model_type = type(model.model).__name__
    model_size = model.model_size()
    first_layers = str(list(model.model_state_dict().keys())[:3])
    identifier = f"{model_type}_{model_size}_{first_layers}"
    final_hash = hashlib.sha256(identifier.encode()).hexdigest()
    logger.debug(f"[MultiGPU_DisTorch_HASH] Created hash for {caller}: {final_hash[:8]}...")
    return final_hash


def register_patched_ggufmodelpatcher():
    """Register and patch the GGUFModelPatcher for distributed loading"""
    from nodes import NODE_CLASS_MAPPINGS
    original_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
    module = sys.modules[original_loader.__module__]

    if not hasattr(module.GGUFModelPatcher, '_patched'):
        original_load = module.GGUFModelPatcher.load

    def new_load(self, *args, force_patch_weights=False, **kwargs):
        global model_allocation_store

        super(module.GGUFModelPatcher, self).load(*args, force_patch_weights=True, **kwargs)
        debug_hash = create_model_hash(self, "patcher")
        linked = []
        module_count = 0
        for n, m in self.model.named_modules():
            module_count += 1
            if hasattr(m, "weight"):
                device = getattr(m.weight, "device", None)
                if device is not None:
                    linked.append((n, m))
                    continue
            if hasattr(m, "bias"):
                device = getattr(m.bias, "device", None)
                if device is not None:
                    linked.append((n, m))
                    continue
        if linked:
            if hasattr(self, 'model'):
                debug_hash = create_model_hash(self, "patcher")
                debug_allocations = model_allocation_store.get(debug_hash)
                if debug_allocations:
                    device_assignments = analyze_ggml_loading(self.model, debug_allocations)['device_assignments']
                    for device, layers in device_assignments.items():
                        target_device = torch.device(device)
                        for n, m, _ in layers:
                            m.to(self.load_device).to(target_device)

                    self.mmap_released = True

    module.GGUFModelPatcher.load = new_load
    module.GGUFModelPatcher._patched = True


def analyze_ggml_loading(model, allocations_str):
    """Analyze and distribute GGML model layers across devices"""
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = allocations_str
    virtual_vram_gb = 0.0

    if '#' in allocations_str:
        distorch_alloc, virtual_vram_str = allocations_str.split('#')
        if not distorch_alloc:
            distorch_alloc = calculate_vvram_allocation_string(model, virtual_vram_str)

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<12}{:>10}{:>14}{:>10}"

    for allocation in distorch_alloc.split(';'):
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
    logger.info("    DisTorch Model Device Allocations")
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

    layer_summary = {}
    layer_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            layer_type = type(module).__name__
            layer_summary[layer_type] = layer_summary.get(layer_type, 0) + 1
            layer_list.append((name, module, layer_type))
            layer_memory = 0
            if module.weight is not None:
                layer_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                layer_memory += module.bias.numel() * module.bias.element_size()
            memory_by_type[layer_type] += layer_memory
            total_memory += layer_memory

    logger.info("    DisTorch Model Layer Distribution")
    logger.info(dash_line)
    fmt_layer = "{:<12}{:>10}{:>14}{:>10}"
    logger.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    for layer_type, count in layer_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_layer.format(layer_type,str(count),f"{mem_mb:.2f}",f"{mem_percent:.1f}%"))
    logger.info(dash_line)

    nonzero_devices = [d for d, r in DEVICE_RATIOS_DISTORCH.items() if r > 0]
    nonzero_total_ratio = sum(DEVICE_RATIOS_DISTORCH[d] for d in nonzero_devices)
    device_assignments = {device: [] for device in DEVICE_RATIOS_DISTORCH.keys()}
    total_layers = len(layer_list)
    current_layer = 0

    for idx, device in enumerate(nonzero_devices):
        ratio = DEVICE_RATIOS_DISTORCH[device]
        if idx == len(nonzero_devices) - 1:
            device_layer_count = total_layers - current_layer
        else:
            device_layer_count = int((ratio / nonzero_total_ratio) * total_layers)
        start_idx = current_layer
        end_idx = current_layer + device_layer_count
        device_assignments[device] = layer_list[start_idx:end_idx]
        current_layer += device_layer_count

    logger.info("DisTorch Model Final Device/Layer Assignments")
    logger.info(dash_line)
    fmt_assign = "{:<12}{:>10}{:>14}{:>10}"
    logger.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)
    total_assigned_memory = 0
    device_memories = {}
    for device, layers in device_assignments.items():
        device_memory = 0
        for layer_type in layer_summary:
            type_layers = sum(1 for _, _, lt in layers if lt == layer_type)
            if layer_summary[layer_type] > 0:
                mem_per_layer = memory_by_type[layer_type] / layer_summary[layer_type]
                device_memory += mem_per_layer * type_layers
        device_memories[device] = device_memory
        total_assigned_memory += device_memory

    sorted_assignments = sorted(device_assignments.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_assignments:
        layers = device_assignments[dev]
        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_assign.format(dev,str(len(layers)),f"{mem_mb:.2f}",f"{mem_percent:.1f}%"))
    logger.info(dash_line)

    return {"device_assignments": device_assignments}


def calculate_vvram_allocation_string(model, virtual_vram_str):
    """Calculate virtual VRAM allocation string for distributed loading"""
    recipient_device, vram_amount, donors = virtual_vram_str.split(';')
    virtual_vram_gb = float(vram_amount)

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<8} {:<6} {:>11} {:>9} {:>9}"

    logger.info(eq_line)
    logger.info("    DisTorch Model Virtual VRAM Analysis")
    logger.info(eq_line)
    logger.info(fmt_assign.format("Object", "Role", "Original(GB)", "Total(GB)", "Virt(GB)"))
    logger.info(dash_line)

    recipient_vram = mm.get_total_memory(torch.device(recipient_device)) / (1024**3)
    recipient_virtual = recipient_vram + virtual_vram_gb

    logger.info(fmt_assign.format(recipient_device, 'recip', f"{recipient_vram:.2f}GB",f"{recipient_virtual:.2f}GB", f"+{virtual_vram_gb:.2f}GB"))

    ram_donors = [d for d in donors.split(',') if d != 'cpu']
    remaining_vram_needed = virtual_vram_gb
    
    donor_device_info = {}
    donor_allocations = {}
    
    for donor in ram_donors:
        donor_vram = mm.get_total_memory(torch.device(donor)) / (1024**3)
        max_donor_capacity = donor_vram * 0.9
        
        donation = min(remaining_vram_needed, max_donor_capacity)
        donor_virtual = donor_vram - donation
        remaining_vram_needed -= donation
        donor_allocations[donor] = donation
            
        donor_device_info[donor] = (donor_vram, donor_virtual)
        logger.info(fmt_assign.format(donor, 'donor', f"{donor_vram:.2f}GB",  f"{donor_virtual:.2f}GB", f"-{donation:.2f}GB"))
    
    system_dram_gb = mm.get_total_memory(torch.device('cpu')) / (1024**3)
    cpu_donation = remaining_vram_needed
    cpu_virtual = system_dram_gb - cpu_donation
    donor_allocations['cpu'] = cpu_donation
    logger.info(fmt_assign.format('cpu', 'donor', f"{system_dram_gb:.2f}GB", f"{cpu_virtual:.2f}GB", f"-{cpu_donation:.2f}GB"))
    
    logger.info(dash_line)

    layer_summary = {}
    layer_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            layer_type = type(module).__name__
            layer_summary[layer_type] = layer_summary.get(layer_type, 0) + 1
            layer_list.append((name, module, layer_type))
            layer_memory = 0
            if module.weight is not None:
                layer_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                layer_memory += module.bias.numel() * module.bias.element_size()
            memory_by_type[layer_type] += layer_memory
            total_memory += layer_memory

    model_size_gb = total_memory / (1024**3)
    new_model_size_gb = max(0, model_size_gb - virtual_vram_gb)

    logger.info(fmt_assign.format('model', 'model', f"{model_size_gb:.2f}GB",f"{new_model_size_gb:.2f}GB", f"-{virtual_vram_gb:.2f}GB"))

    if model_size_gb > (recipient_vram * 0.9):
        on_recipient = recipient_vram * 0.9
        on_virtuals = model_size_gb - on_recipient
        logger.info(f"\nWarning: Model size is greater than 90% of recipient VRAM. {on_virtuals:.2f} GB of GGML Layers Offloaded Automatically to Virtual VRAM.\n")
    else:
        on_recipient = model_size_gb
        on_virtuals = 0

    new_on_recipient = max(0, on_recipient - virtual_vram_gb)

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
    logger.info(fmt_mem.format("\n  v1 Expert String", allocation_string))

    return allocation_string


def override_class_with_distorch_gguf(cls):
    """Legacy DisTorch wrapper for GGUF models for backward compatibility."""
    from .nodes import get_device_list
    from . import current_device
    
    class NodeOverrideDisTorchGGUFLegacy(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.1})
            inputs["optional"]["use_other_vram"] = ("BOOLEAN", {"default": False})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {
                "multiline": False, 
                "default": "",
            })
            return inputs

        CATEGORY = "multigpu/legacy"
        FUNCTION = "override"
        if hasattr(cls, 'TITLE'):
            TITLE = f"{cls.TITLE} (Legacy)"
        else:
            TITLE = "Legacy DisTorch Node"

        def override(self, *args, device=None, expert_mode_allocations=None, use_other_vram=None, virtual_vram_gb=0.0, **kwargs):
            from . import set_current_device
            if device is not None:
                set_current_device(device)
            
            register_patched_ggufmodelpatcher()
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)

            vram_string = ""
            if virtual_vram_gb > 0:
                if use_other_vram:
                    available_devices = [d for d in get_device_list() if d.startswith(("cuda", "xpu"))]
                    other_devices = [d for d in available_devices if d != device]
                    other_devices.sort(key=lambda x: int(x.split(':')[1] if ':' in x else x[-1]), reverse=False)
                    device_string = ','.join(other_devices + ['cpu'])
                    vram_string = f"{device};{virtual_vram_gb};{device_string}"
                else:
                    vram_string = f"{device};{virtual_vram_gb};cpu"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            if hasattr(out[0], 'model'):
                model_hash = create_model_hash(out[0], "override")
                model_allocation_store[model_hash] = full_allocation
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_model_hash(out[0].patcher, "override")
                model_allocation_store[model_hash] = full_allocation

            return out

    return NodeOverrideDisTorchGGUFLegacy


def override_class_with_distorch_gguf_v2(cls):
    """DisTorch 2.0 wrapper for GGUF models."""
    from .nodes import get_device_list
    from . import current_device
    
    class NodeOverrideDisTorchGGUFv2(cls):
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
            if compute_device is not None:
                set_current_device(compute_device)
            
            register_patched_ggufmodelpatcher()
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)

            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{compute_device};{virtual_vram_gb};{donor_device}"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            logger.info(f"[MultiGPU_DisTorch] Full allocation string: {full_allocation}")
            
            if hasattr(out[0], 'model'):
                model_hash = create_model_hash(out[0], "override")
                model_allocation_store[model_hash] = full_allocation
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_model_hash(out[0].patcher, "override")
                model_allocation_store[model_hash] = full_allocation

            return out

    return NodeOverrideDisTorchGGUFv2


def override_class_with_distorch_clip(cls):
    """DisTorch wrapper for CLIP models with GGUF support"""
    from .nodes import get_device_list
    from . import current_text_encoder_device
    
    class NodeOverrideDisTorch(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.1})
            inputs["optional"]["use_other_vram"] = ("BOOLEAN", {"default": False})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {
                "multiline": False, 
                "default": "",
                "tooltip": "Expert use only: Manual VRAM allocation string. Incorrect values can cause crashes. Do not modify unless you fully understand DisTorch memory management."
            })
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, expert_mode_allocations=None, use_other_vram=None, virtual_vram_gb=0.0, **kwargs):
            from . import set_current_text_encoder_device
            if device is not None:
                set_current_text_encoder_device(device)
            
            register_patched_ggufmodelpatcher()
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)

            vram_string = ""
            if virtual_vram_gb > 0:
                if use_other_vram:
                    available_devices = [d for d in get_device_list() if d.startswith(("cuda", "xpu"))]
                    other_devices = [d for d in available_devices if d != device]
                    other_devices.sort(key=lambda x: int(x.split(':')[1] if ':' in x else x[-1]), reverse=False)
                    device_string = ','.join(other_devices + ['cpu'])
                    vram_string = f"{device};{virtual_vram_gb};{device_string}"
                else:
                    vram_string = f"{device};{virtual_vram_gb};cpu"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            logging.info(f"[MultiGPU_DisTorch] Full allocation string: {full_allocation}")
            
            if hasattr(out[0], 'model'):
                model_hash = create_model_hash(out[0], "override")
                model_allocation_store[model_hash] = full_allocation
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_model_hash(out[0].patcher, "override")
                model_allocation_store[model_hash] = full_allocation

            return out

    return NodeOverrideDisTorch


# Alias for backward compatibility
override_class_with_distorch = override_class_with_distorch_gguf
