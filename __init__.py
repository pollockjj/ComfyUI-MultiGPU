import copy
import torch
import sys
import comfy.model_management as mm
import os
from pathlib import Path
import logging
import folder_paths
import shutil
from collections import defaultdict
import hashlib
import tempfile
import subprocess
import gc
from safetensors.torch import save_file, load_file
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
    WanVideoModelLoader, WanVideoVAELoader, LoadWanVideoT5TextEncoder
)

current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()
model_allocation_store = {}

def get_torch_device_patched():
    device = None
    if (not torch.cuda.is_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        device = torch.device(current_device)
    return device

def text_encoder_device_patched():
    device = None
    if (not torch.cuda.is_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        device = torch.device(current_text_encoder_device)
    return device

mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched


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
    import torch
    return ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]

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


class MergeFluxLoRAsQuantizeAndLoad:
    @classmethod
    def INPUT_TYPES(cls):
        unet_name = folder_paths.get_filename_list("diffusion_models")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "unet_name": (unet_name,),
                "switch_1": (["Off", "On"],),
                "lora_name_1": (loras,),
                "lora_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_2": (["Off", "On"],),
                "lora_name_2": (loras,),
                "lora_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_3": (["Off", "On"],),
                "lora_name_3": (loras,),
                "lora_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_4": (["Off", "On"],),
                "lora_name_4": (loras,),
                "lora_weight_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "quantization": (["Q2_K", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_S", "Q6_K", "Q8_0", "FP16"], {"default": "Q4_K_S"}),
                "delete_final_gguf": ("BOOLEAN", {"default": False}),
                "new_model_name": ("STRING", {"default": "merged_model"}),
            }
        }
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_and_quantize"
    CATEGORY = "loaders"

    def merge_flux_loras(self, model_sd: dict, lora_paths: list, weights: list, device="cuda") -> dict:
        for lora_path, weight in zip(lora_paths, weights):
            logging.info(f"[DEBUG] Merging LoRA file: {lora_path} with weight: {weight}")
            lora_sd = load_file(lora_path, device=device)
            for key in list(lora_sd.keys()):
                if "lora_down" not in key:
                    continue
                base_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                module_name = base_name.replace("_", ".")
                alpha_key = f"{base_name}.alpha"
                if module_name not in model_sd:
                    logging.info(f"[DEBUG] Module {module_name} not found in model_sd; skipping key {key}")
                    continue
                down_weight = lora_sd[key].float()
                up_weight = lora_sd[up_key].float()
                alpha = float(lora_sd.get(alpha_key, up_weight.shape[0]))
                scale = weight * alpha / up_weight.shape[0]
                logging.info(f"[DEBUG] Merging module: {module_name} with alpha: {alpha}, scale: {scale}")
                target_weight = model_sd[module_name]
                if len(target_weight.shape) == 2:
                    update = (up_weight @ down_weight) * scale
                else:
                    if down_weight.shape[2:4] == (1, 1):
                        update = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                        update = update.unsqueeze(2).unsqueeze(3) * scale
                    else:
                        update = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3) * scale
                model_sd[module_name] = target_weight + update.to(target_weight.dtype)
                logging.info(f"[DEBUG] Updated module: {module_name}")
                del up_weight, down_weight, update
            del lora_sd
            torch.cuda.empty_cache()
        return model_sd

    def convert_to_gguf(self, model_path, working_dir):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        convert_script = os.path.join(base_path, "ComfyUI-GGUF", "tools", "convert.py")
        temp_gguf = os.path.join(working_dir, "temp_converted.gguf")
        logging.info("[DEBUG] Running conversion script: " + convert_script)
        subprocess.run([sys.executable, convert_script, "--src", model_path, "--dst", temp_gguf], check=True)
        logging.info("[DEBUG] Conversion complete.")
        return temp_gguf

    def load_and_quantize(self, unet_name, quantization, delete_final_gguf, new_model_name, **kwargs):
        mapping = {"FP16": "F16"}
        logging.info(f"[DEBUG] Starting load_and_quantize: {new_model_name} | Quantization: {quantization}")
        with tempfile.TemporaryDirectory() as merge_dir:
            merged_model_path = os.path.join(merge_dir, "merged_model.safetensors")
            model_path = folder_paths.get_full_path("diffusion_models", unet_name)
            lora_list = []
            for i in range(1, 5):
                name = kwargs.get(f"lora_name_{i}", "None")
                switch = kwargs.get(f"switch_{i}", "Off")
                logging.info(f"[DEBUG] Processing LoRA slot {i}: name = {name}, switch = {switch}")
                if switch == "On" and name and name != "None":
                    lora_file_path = folder_paths.get_full_path("loras", name)
                    weight = kwargs.get(f"lora_weight_{i}", 1.0)
                    lora_list.append((lora_file_path, weight))
                    logging.info(f"[DEBUG] Slot {i} active: path = {lora_file_path}, weight = {weight}")
                else:
                    logging.info(f"[DEBUG] Slot {i} is inactive")
            logging.info(f"[DEBUG] Total active LoRAs: {len(lora_list)}")
            if lora_list:
                model_sd = load_file(model_path, device="cuda")
                model_sd = self.merge_flux_loras(
                    model_sd,
                    [lp for lp, _ in lora_list],
                    [w for _, w in lora_list]
                )
                save_file(model_sd, merged_model_path)
                del model_sd
                torch.cuda.empty_cache()
            else:
                shutil.copy2(model_path, merged_model_path)
            initial_gguf = self.convert_to_gguf(merged_model_path, merge_dir)
            logging.info("[DEBUG] Initial GGUF file created.")
            if quantization == "FP16":
                final_gguf = os.path.join(merge_dir, f"{new_model_name}-{mapping.get(quantization, quantization)}.gguf")
                shutil.copy2(initial_gguf, final_gguf)
                logging.info("[DEBUG] FP16 selected; conversion skipped.")
            else:
                binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binaries", "linux", "llama-quantize")
                final_gguf = os.path.join(merge_dir, f"quantized_{quantization}.gguf")
                subprocess.run([binary, initial_gguf, final_gguf, quantization], check=True)
                logging.info("[DEBUG] Quantization completed.")
            models_dir = os.path.join(folder_paths.models_dir, "unet")
            os.makedirs(models_dir, exist_ok=True)
            final_name = f"{new_model_name}-{mapping.get(quantization, quantization)}.gguf"
            final_path = os.path.join(models_dir, final_name)
            shutil.copy2(final_gguf, final_path)
            logging.info("[DEBUG] Final model file copied to: " + final_path)
            logging.info("[DEBUG] Loading final model.")
            loader = UnetLoaderGGUF()
            result = loader.load_unet(final_name)
            logging.info("[DEBUG] Final model loaded.")
            if delete_final_gguf:
                os.unlink(final_path)
            return result


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
            if device is not None:
                current_device = device
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
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
                    available_devices = [d for d in get_device_list() if d.startswith('cuda')]
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
                    available_devices = [d for d in get_device_list() if d.startswith('cuda')]
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

NODE_CLASS_MAPPINGS["MergeFluxLoRAsQuantizeAndLoaddMultiGPU"] = override_class(MergeFluxLoRAsQuantizeAndLoad)

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
    NODE_CLASS_MAPPINGS["WanVideoModelLoaderMultiGPU"] = override_class(WanVideoModelLoader)
    NODE_CLASS_MAPPINGS["WanVideoVAELoaderMultiGPU"] = override_class(WanVideoVAELoader)
    NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoderMultiGPU"] = override_class(LoadWanVideoT5TextEncoder)


logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
