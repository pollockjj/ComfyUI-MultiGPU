import copy
import torch
import sys
import comfy.model_management as mm
import os
from pathlib import Path
import logging
import folder_paths
from collections import defaultdict
import hashlib
import comfy.utils
from typing import Dict, List 


from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
from .nodes import (
    UnetLoaderGGUF, UnetLoaderGGUFAdvanced,
    CLIPLoaderGGUF, DualCLIPLoaderGGUF, TripleCLIPLoaderGGUF, QuadrupleCLIPLoaderGGUF,
    LTXVLoader,
    Florence2ModelLoader, DownloadAndLoadFlorence2Model,
    CheckpointLoaderNF4,
    LoadFluxControlNet,
    MMAudioModelLoader, MMAudioFeatureUtilsLoader, MMAudioSampler,
    PulidModelLoader, PulidInsightFaceLoader, PulidEvaClipLoader,
    HyVideoModelLoader, HyVideoVAELoader, DownloadAndLoadHyVideoTextEncoder,
    WanVideoModelLoader, WanVideoModelLoader_2, WanVideoVAELoader, LoadWanVideoT5TextEncoder, LoadWanVideoClipTextEncoder,
    WanVideoTextEncode, WanVideoBlockSwap, WanVideoSampler
)

current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()
model_allocation_store = {}

def _has_xpu():
    try:
        return hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available()
    except Exception:
        return False

def get_torch_device_patched():
    device = None
    if (not (torch.cuda.is_available() or _has_xpu()) or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    logging.info(f"[MultiGPU get_torch_device_patched] Returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    device = None
    if (not (torch.cuda.is_available() or _has_xpu()) or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    logging.info(f"[MultiGPU text_encoder_device_patched] Returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device

logging.info(f"[MultiGPU] Patching mm.get_torch_device and mm.text_encoder_device")
logging.info(f"[MultiGPU] Initial current_device: {current_device}")
logging.info(f"[MultiGPU] Initial current_text_encoder_device: {current_text_encoder_device}")
mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched
logging.info(f"[MultiGPU] Patches applied successfully")


def create_model_hash(model, caller):

   model_type = type(model.model).__name__
   model_size = model.model_size()
   first_layers = str(list(model.model_state_dict().keys())[:3])
   identifier = f"{model_type}_{model_size}_{first_layers}"
   final_hash = hashlib.sha256(identifier.encode()).hexdigest()
 
   return final_hash

def register_patched_ggufmodelpatcher():
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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(eq_line)
    logging.info("          DisTorch Device Allocations")
    logging.info(eq_line)
    logging.info(fmt_assign.format("Device", "Alloc %", "Total (GB)", " Alloc (GB)"))
    logging.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_devices:
        frac = device_table[dev]["fraction"]
        tot_gb = device_table[dev]["total_gb"]
        alloc_gb = device_table[dev]["alloc_gb"]
        logging.info(fmt_assign.format(dev,f"{int(frac * 100)}%",f"{tot_gb:.2f}",f"{alloc_gb:.2f}"))

    logging.info(dash_line)

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

    logging.info("     DisTorch GGML Layer Distribution")
    logging.info(dash_line)
    fmt_layer = "{:<12}{:>10}{:>14}{:>10}"
    logging.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logging.info(dash_line)
    for layer_type, count in layer_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logging.info(fmt_layer.format(layer_type,str(count),f"{mem_mb:.2f}",f"{mem_percent:.1f}%"))
    logging.info(dash_line)

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

    logging.info("    DisTorch Final Device/Layer Assignments")
    logging.info(dash_line)
    fmt_assign = "{:<12}{:>10}{:>14}{:>10}"
    logging.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logging.info(dash_line)
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
        logging.info(fmt_assign.format(dev,str(len(layers)),f"{mem_mb:.2f}",f"{mem_percent:.1f}%"))
    logging.info(dash_line)

    return {"device_assignments": device_assignments}

def calculate_vvram_allocation_string(model, virtual_vram_str):
    recipient_device, vram_amount, donors = virtual_vram_str.split(';')
    virtual_vram_gb = float(vram_amount)

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<8} {:<6} {:>11} {:>9} {:>9}"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(eq_line)
    logging.info("          DisTorch Virtual VRAM Analysis")
    logging.info(eq_line)
    logging.info(fmt_assign.format("Object", "Role", "Original(GB)", "Total(GB)", "Virt(GB)"))
    logging.info(dash_line)

    recipient_vram = mm.get_total_memory(torch.device(recipient_device)) / (1024**3)
    recipient_virtual = recipient_vram + virtual_vram_gb

    logging.info(fmt_assign.format(recipient_device, 'recip', f"{recipient_vram:.2f}GB",f"{recipient_virtual:.2f}GB", f"+{virtual_vram_gb:.2f}GB"))

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
        logging.info(fmt_assign.format(donor, 'donor', f"{donor_vram:.2f}GB",  f"{donor_virtual:.2f}GB", f"-{donation:.2f}GB"))
    
    system_dram_gb = mm.get_total_memory(torch.device('cpu')) / (1024**3)
    cpu_donation = remaining_vram_needed
    cpu_virtual = system_dram_gb - cpu_donation
    donor_allocations['cpu'] = cpu_donation
    logging.info(fmt_assign.format('cpu', 'donor', f"{system_dram_gb:.2f}GB", f"{cpu_virtual:.2f}GB", f"-{cpu_donation:.2f}GB"))
    
    logging.info(dash_line)

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

    logging.info(fmt_assign.format('model', 'model', f"{model_size_gb:.2f}GB",f"{new_model_size_gb:.2f}GB", f"-{virtual_vram_gb:.2f}GB"))

    if model_size_gb > (recipient_vram * 0.9):
        on_recipient = recipient_vram * 0.9
        on_virtuals = model_size_gb - on_recipient
        logging.info(f"\nWarning: Model size is greater than 90% of recipient VRAM. {on_virtuals:.2f} GB of GGML Layers Offloaded Automatically to Virtual VRAM.\n")
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
    logging.info(fmt_mem.format("\nAllocation String", allocation_string))

    return allocation_string

def get_device_list():
    devs = ["cpu"]
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if _has_xpu():
            devs += [f"xpu:{i}" for i in range(torch.xpu.device_count())]
    except Exception:
        pass
    return devs

class DeviceSelectorMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        return {
            "required": {
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0]})
            }
        }

    RETURN_TYPES = (get_device_list(),)
    RETURN_NAMES = ("device",)
    FUNCTION = "select_device"
    CATEGORY = "multigpu"

    def select_device(self, device):
        return (device,)

class HunyuanVideoEmbeddingsAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hyvid_embeds": ("HYVIDEMBEDS",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "adapt_embeddings"
    CATEGORY = "multigpu"

    def adapt_embeddings(self, hyvid_embeds):
        cond = hyvid_embeds["prompt_embeds"]
        
        pooled_dict = {
            "pooled_output": hyvid_embeds["prompt_embeds_2"],
            "cross_attn": hyvid_embeds["prompt_embeds"],
            "attention_mask": hyvid_embeds["attention_mask"],
        }
        
        if hyvid_embeds["attention_mask_2"] is not None:
            pooled_dict["attention_mask_controlnet"] = hyvid_embeds["attention_mask_2"]

        if hyvid_embeds["cfg"] is not None:
            pooled_dict["guidance"] = float(hyvid_embeds["cfg"])
            pooled_dict["start_percent"] = float(hyvid_embeds["start_percent"]) if hyvid_embeds["start_percent"] is not None else 0.0
            pooled_dict["end_percent"] = float(hyvid_embeds["end_percent"]) if hyvid_embeds["end_percent"] is not None else 1.0

        return ([[cond, pooled_dict]],)




def override_class(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            global current_device
            
            logging.info(f"[MultiGPU override_class] Called with device={device}, current_device={current_device}")
            
            if device is not None:
                current_device = device
                logging.info(f"[MultiGPU override_class] Setting current_device to {device}")
            
            fn = getattr(super(), cls.FUNCTION)
            logging.info(f"[MultiGPU override_class] Calling wrapped function: {cls.__name__}.{cls.FUNCTION}")
            out = fn(*args, **kwargs)
            logging.info(f"[MultiGPU override_class] Wrapped function completed successfully")
            
            return out

    return NodeOverride

def override_class_clip(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            global current_text_encoder_device
            
            if device is not None:
                current_text_encoder_device = device
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            return out

    return NodeOverride

def override_class_with_distorch(cls):
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
            global current_device
            if device is not None:
                current_device = device
            
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
            
            logging.info(f"[DisTorch] Full allocation string: {full_allocation}")
            
            if hasattr(out[0], 'model'):
                model_hash = create_model_hash(out[0], "override")
                model_allocation_store[model_hash] = full_allocation
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_model_hash(out[0].patcher, "override")
                model_allocation_store[model_hash] = full_allocation

            return out

    return NodeOverrideDisTorch

def override_class_with_distorch_clip(cls):
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
            global current_text_encoder_device
            if device is not None:
                current_text_encoder_device = device
            
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
            
            logging.info(f"[DisTorch] Full allocation string: {full_allocation}")
            
            if hasattr(out[0], 'model'):
                model_hash = create_model_hash(out[0], "override")
                model_allocation_store[model_hash] = full_allocation
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_hash = create_model_hash(out[0].patcher, "override")
                model_allocation_store[model_hash] = full_allocation

            return out

    return NodeOverrideDisTorch

def check_module_exists(module_path):
    full_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], module_path)
    logging.info(f"MultiGPU: Checking for module at {full_path}")
    if not os.path.exists(full_path):
        logging.info(f"MultiGPU: Module {module_path} not found - skipping")
        return False
    logging.info(f"MultiGPU: Found {module_path}, creating compatible MultiGPU nodes")
    return True

NODE_CLASS_MAPPINGS = {
    "DeviceSelectorMultiGPU": DeviceSelectorMultiGPU,
    "HunyuanVideoEmbeddingsAdapter": HunyuanVideoEmbeddingsAdapter,
}


NODE_CLASS_MAPPINGS["UNETLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
NODE_CLASS_MAPPINGS["TripleCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])

if check_module_exists("ComfyUI-LTXVideo") or check_module_exists("comfyui-ltxvideo"):
    NODE_CLASS_MAPPINGS["LTXVLoaderMultiGPU"] = override_class(LTXVLoader)

if check_module_exists("ComfyUI-Florence2") or check_module_exists("comfyui-florence2"):
    NODE_CLASS_MAPPINGS["Florence2ModelLoaderMultiGPU"] = override_class(Florence2ModelLoader)
    NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2ModelMultiGPU"] = override_class(DownloadAndLoadFlorence2Model)

if check_module_exists("ComfyUI_bitsandbytes_NF4") or check_module_exists("comfyui_bitsandbytes_nf4"):
    NODE_CLASS_MAPPINGS["CheckpointLoaderNF4MultiGPU"] = override_class(CheckpointLoaderNF4)

if check_module_exists("x-flux-comfyui") or check_module_exists("x-flux-comfyui"):
    NODE_CLASS_MAPPINGS["LoadFluxControlNetMultiGPU"] = override_class(LoadFluxControlNet)

if check_module_exists("ComfyUI-MMAudio") or check_module_exists("comfyui-mmaudio"):
    NODE_CLASS_MAPPINGS["MMAudioModelLoaderMultiGPU"] = override_class(MMAudioModelLoader)
    NODE_CLASS_MAPPINGS["MMAudioFeatureUtilsLoaderMultiGPU"] = override_class(MMAudioFeatureUtilsLoader)
    NODE_CLASS_MAPPINGS["MMAudioSamplerMultiGPU"] = override_class(MMAudioSampler)

if check_module_exists("ComfyUI-GGUF") or check_module_exists("comfyui-gguf"):
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFMultiGPU"] = override_class(UnetLoaderGGUF)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch(UnetLoaderGGUF)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvancedMultiGPU"] = override_class(UnetLoaderGGUFAdvanced)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvancedDisTorchMultiGPU"] = override_class_with_distorch(UnetLoaderGGUFAdvanced)
    NODE_CLASS_MAPPINGS["CLIPLoaderGGUFMultiGPU"] = override_class_clip(CLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["CLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(CLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUFMultiGPU"] = override_class_clip(DualCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(DualCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUFMultiGPU"] = override_class_clip(TripleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(TripleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUFMultiGPU"] = override_class_clip(QuadrupleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(QuadrupleCLIPLoaderGGUF)

if check_module_exists("PuLID_ComfyUI") or check_module_exists("pulid_comfyui"):
    NODE_CLASS_MAPPINGS["PulidModelLoaderMultiGPU"] = override_class(PulidModelLoader)
    NODE_CLASS_MAPPINGS["PulidInsightFaceLoaderMultiGPU"] = override_class(PulidInsightFaceLoader)
    NODE_CLASS_MAPPINGS["PulidEvaClipLoaderMultiGPU"] = override_class(PulidEvaClipLoader)

if check_module_exists("ComfyUI-HunyuanVideoWrapper") or check_module_exists("comfyui-hunyuanvideowrapper"):
    NODE_CLASS_MAPPINGS["HyVideoModelLoaderMultiGPU"] = override_class(HyVideoModelLoader)
    NODE_CLASS_MAPPINGS["HyVideoVAELoaderMultiGPU"] = override_class(HyVideoVAELoader)
    NODE_CLASS_MAPPINGS["DownloadAndLoadHyVideoTextEncoderMultiGPU"] = override_class(DownloadAndLoadHyVideoTextEncoder)

if check_module_exists("ComfyUI-WanVideoWrapper") or check_module_exists("comfyui-wanvideowrapper"):
    # WanVideo uses custom implementation, not the standard override
    NODE_CLASS_MAPPINGS["WanVideoModelLoaderMultiGPU"] = WanVideoModelLoader
    NODE_CLASS_MAPPINGS["WanVideoModelLoaderMultiGPU_2"] = WanVideoModelLoader_2
    NODE_CLASS_MAPPINGS["WanVideoVAELoaderMultiGPU"] = WanVideoVAELoader
    NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoderMultiGPU"] = LoadWanVideoT5TextEncoder
    NODE_CLASS_MAPPINGS["LoadWanVideoClipTextEncoderMultiGPU"] = LoadWanVideoClipTextEncoder
    NODE_CLASS_MAPPINGS["WanVideoTextEncodeMultiGPU"] = WanVideoTextEncode
    NODE_CLASS_MAPPINGS["WanVideoBlockSwapMultiGPU"] = WanVideoBlockSwap
    NODE_CLASS_MAPPINGS["WanVideoSamplerMultiGPU"] = WanVideoSampler


logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
