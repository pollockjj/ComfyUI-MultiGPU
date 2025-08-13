import torch
import logging
import os
import copy
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS

# Global device state management
current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()

def _has_xpu():
    try:
        return hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available()
    except Exception:
        return False

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

def set_current_device(device):
    global current_device
    current_device = device
    logging.info(f"[MultiGPU] current_device set to: {device}")

def set_current_text_encoder_device(device):
    global current_text_encoder_device
    current_text_encoder_device = device
    logging.info(f"[MultiGPU] current_text_encoder_device set to: {device}")

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
            logging.info(f"[MultiGPU override_class] Called with device={device}")
            
            if device is not None:
                set_current_device(device)
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
            if device is not None:
                set_current_text_encoder_device(device)
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            return out

    return NodeOverride

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

# Apply patches
logging.info(f"[MultiGPU] Patching mm.get_torch_device and mm.text_encoder_device")
logging.info(f"[MultiGPU] Initial current_device: {current_device}")
logging.info(f"[MultiGPU] Initial current_text_encoder_device: {current_text_encoder_device}")
mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched
logging.info(f"[MultiGPU] Patches applied successfully")

def check_module_exists(module_path):
    full_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], module_path)
    logging.info(f"MultiGPU: Checking for module at {full_path}")
    if not os.path.exists(full_path):
        logging.info(f"MultiGPU: Module {module_path} not found - skipping")
        return False
    logging.info(f"MultiGPU: Found {module_path}, creating compatible MultiGPU nodes")
    return True

# Import from nodes.py
from .nodes import (
    DeviceSelectorMultiGPU,
    HunyuanVideoEmbeddingsAdapter,
    UnetLoaderGGUF,
    UnetLoaderGGUFAdvanced,
    CLIPLoaderGGUF,
    DualCLIPLoaderGGUF,
    TripleCLIPLoaderGGUF,
    QuadrupleCLIPLoaderGGUF,
    LTXVLoader,
    Florence2ModelLoader,
    DownloadAndLoadFlorence2Model,
    CheckpointLoaderNF4,
    LoadFluxControlNet,
    MMAudioModelLoader,
    MMAudioFeatureUtilsLoader,
    MMAudioSampler,
    PulidModelLoader,
    PulidInsightFaceLoader,
    PulidEvaClipLoader,
    HyVideoModelLoader,
    HyVideoVAELoader,
    DownloadAndLoadHyVideoTextEncoder,
)

# Import from wanvideo.py
from .wanvideo import (
    WanVideoModelLoader,
    WanVideoModelLoader_2,
    WanVideoVAELoader,
    LoadWanVideoT5TextEncoder,
    LoadWanVideoClipTextEncoder,
    WanVideoTextEncode,
    WanVideoBlockSwap,
    WanVideoSampler
)

# Import from distorch.py
from .distorch import (
    model_allocation_store,
    create_model_hash,
    register_patched_ggufmodelpatcher,
    analyze_ggml_loading,
    calculate_vvram_allocation_string,
    override_class_with_distorch_gguf,
    override_class_with_distorch_gguf_v2,
    override_class_with_distorch_clip,
    override_class_with_distorch
)

# Import from block_swap.py
from .block_swap import (
    analyze_safetensor_distorch,
    apply_block_swap,
    override_class_with_distorch_safetensor,
    override_class_with_distorch_bs
)

# Import from distorch_safetensor.py for FLUX support
from .distorch_safetensor import (
    safetensor_allocation_store,
    create_safetensor_model_hash,
    register_patched_safetensor_modelpatcher,
    analyze_safetensor_loading,
    calculate_safetensor_vvram_allocation,
    override_class_with_distorch_safetensor_v2
)

# Initialize NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "DeviceSelectorMultiGPU": DeviceSelectorMultiGPU,
    "HunyuanVideoEmbeddingsAdapter": HunyuanVideoEmbeddingsAdapter,
}

# Standard MultiGPU nodes
NODE_CLASS_MAPPINGS["UNETLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
if "QuadrupleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
if "DiffusersLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffusersLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
if "DiffControlNetLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

# DisTorch 2 SafeTensor nodes for FLUX and other safetensor models
NODE_CLASS_MAPPINGS["UNETLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
if "QuadrupleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
if "DiffusersLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffusersLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
if "DiffControlNetLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

# DisTorch 2 FLUX-specific nodes (these are the ones users will use for FLUX)
logging.info("[DISTORCH_SAFETENSOR] Registering FLUX DisTorch2 nodes")
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleFLUXDisTorch2"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["UNETLoaderFLUXDisTorch2"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
if "DualCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DualCLIPLoaderFLUXDisTorch2"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderFLUXDisTorch2"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])

# ComfyUI-LTXVideo
if check_module_exists("ComfyUI-LTXVideo") or check_module_exists("comfyui-ltxvideo"):
    NODE_CLASS_MAPPINGS["LTXVLoaderMultiGPU"] = override_class(LTXVLoader)

# ComfyUI-Florence2
if check_module_exists("ComfyUI-Florence2") or check_module_exists("comfyui-florence2"):
    NODE_CLASS_MAPPINGS["Florence2ModelLoaderMultiGPU"] = override_class(Florence2ModelLoader)
    NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2ModelMultiGPU"] = override_class(DownloadAndLoadFlorence2Model)

# ComfyUI_bitsandbytes_NF4
if check_module_exists("ComfyUI_bitsandbytes_NF4") or check_module_exists("comfyui_bitsandbytes_nf4"):
    NODE_CLASS_MAPPINGS["CheckpointLoaderNF4MultiGPU"] = override_class(CheckpointLoaderNF4)

# x-flux-comfyui
if check_module_exists("x-flux-comfyui") or check_module_exists("x-flux-comfyui"):
    NODE_CLASS_MAPPINGS["LoadFluxControlNetMultiGPU"] = override_class(LoadFluxControlNet)

# ComfyUI-MMAudio
if check_module_exists("ComfyUI-MMAudio") or check_module_exists("comfyui-mmaudio"):
    NODE_CLASS_MAPPINGS["MMAudioModelLoaderMultiGPU"] = override_class(MMAudioModelLoader)
    NODE_CLASS_MAPPINGS["MMAudioFeatureUtilsLoaderMultiGPU"] = override_class(MMAudioFeatureUtilsLoader)
    NODE_CLASS_MAPPINGS["MMAudioSamplerMultiGPU"] = override_class(MMAudioSampler)

# ComfyUI-GGUF
if check_module_exists("ComfyUI-GGUF") or check_module_exists("comfyui-gguf"):
    # Legacy DisTorch GGUF nodes
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_gguf(UnetLoaderGGUF)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvancedDisTorchMultiGPU"] = override_class_with_distorch_gguf(UnetLoaderGGUFAdvanced)
    NODE_CLASS_MAPPINGS["CLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(CLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(DualCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(TripleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_clip(QuadrupleCLIPLoaderGGUF)

    # DisTorch 2 GGUF nodes
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorch2MultiGPU"] = override_class_with_distorch_gguf_v2(UnetLoaderGGUF)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvancedDisTorch2MultiGPU"] = override_class_with_distorch_gguf_v2(UnetLoaderGGUFAdvanced)
    NODE_CLASS_MAPPINGS["CLIPLoaderGGUFDisTorch2MultiGPU"] = override_class_with_distorch_clip(CLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUFDisTorch2MultiGPU"] = override_class_with_distorch_clip(DualCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUFDisTorch2MultiGPU"] = override_class_with_distorch_clip(TripleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUFDisTorch2MultiGPU"] = override_class_with_distorch_clip(QuadrupleCLIPLoaderGGUF)

    # Standard MultiGPU nodes
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFMultiGPU"] = override_class(UnetLoaderGGUF)
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvancedMultiGPU"] = override_class(UnetLoaderGGUFAdvanced)
    NODE_CLASS_MAPPINGS["CLIPLoaderGGUFMultiGPU"] = override_class_clip(CLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUFMultiGPU"] = override_class_clip(DualCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUFMultiGPU"] = override_class_clip(TripleCLIPLoaderGGUF)
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUFMultiGPU"] = override_class_clip(QuadrupleCLIPLoaderGGUF)

# PuLID_ComfyUI
if check_module_exists("PuLID_ComfyUI") or check_module_exists("pulid_comfyui"):
    NODE_CLASS_MAPPINGS["PulidModelLoaderMultiGPU"] = override_class(PulidModelLoader)
    NODE_CLASS_MAPPINGS["PulidInsightFaceLoaderMultiGPU"] = override_class(PulidInsightFaceLoader)
    NODE_CLASS_MAPPINGS["PulidEvaClipLoaderMultiGPU"] = override_class(PulidEvaClipLoader)

# ComfyUI-HunyuanVideoWrapper
if check_module_exists("ComfyUI-HunyuanVideoWrapper") or check_module_exists("comfyui-hunyuanvideowrapper"):
    NODE_CLASS_MAPPINGS["HyVideoModelLoaderMultiGPU"] = override_class(HyVideoModelLoader)
    NODE_CLASS_MAPPINGS["HyVideoVAELoaderMultiGPU"] = override_class(HyVideoVAELoader)
    NODE_CLASS_MAPPINGS["DownloadAndLoadHyVideoTextEncoderMultiGPU"] = override_class(DownloadAndLoadHyVideoTextEncoder)

# ComfyUI-WanVideoWrapper
if check_module_exists("ComfyUI-WanVideoWrapper") or check_module_exists("comfyui-wanvideowrapper"):
    NODE_CLASS_MAPPINGS["WanVideoModelLoaderMultiGPU"] = WanVideoModelLoader
    NODE_CLASS_MAPPINGS["WanVideoModelLoaderMultiGPU_2"] = WanVideoModelLoader_2
    NODE_CLASS_MAPPINGS["WanVideoVAELoaderMultiGPU"] = WanVideoVAELoader
    NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoderMultiGPU"] = LoadWanVideoT5TextEncoder
    NODE_CLASS_MAPPINGS["LoadWanVideoClipTextEncoderMultiGPU"] = LoadWanVideoClipTextEncoder
    NODE_CLASS_MAPPINGS["WanVideoTextEncodeMultiGPU"] = WanVideoTextEncode
    NODE_CLASS_MAPPINGS["WanVideoBlockSwapMultiGPU"] = WanVideoBlockSwap
    NODE_CLASS_MAPPINGS["WanVideoSamplerMultiGPU"] = WanVideoSampler

logging.info(f"MultiGPU: Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")

# --- Memory Logging Test ---
from .debug_utils import log_memory_usage
log_memory_usage("ComfyUI Startup")
