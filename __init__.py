import torch
import logging
import weakref
import os
import copy
from pathlib import Path
import folder_paths
import comfy.model_management as mm
import comfy.model_patcher
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
from .device_utils import (
    get_device_list,
    is_accelerator_available,
    soft_empty_cache_multigpu,
)
from .model_management_mgpu import (
    trigger_executor_cache_reset,
    check_cpu_memory_threshold,
    multigpu_memory_log,
    force_full_system_cleanup,
)

WEB_DIRECTORY = "./web"
MGPU_MM_LOG = False
DEBUG_LOG = False

logger = logging.getLogger("MultiGPU")
logger.propagate = False

if not logger.handlers:
    log_level = logging.DEBUG if DEBUG_LOG else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

def mgpu_mm_log_method(self, msg):
    """Add MultiGPU model management logging method to logger instance."""
    if MGPU_MM_LOG:
        self.info(f"[MultiGPU Model Management] {msg}")
logger.mgpu_mm_log = mgpu_mm_log_method.__get__(logger, type(logger))

def check_module_exists(module_path):
    """Check if a custom node module exists in ComfyUI custom_nodes directory."""
    full_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], module_path)
    logger.debug(f"[MultiGPU] Checking for module at {full_path}")
    if not os.path.exists(full_path):
        logger.debug(f"[MultiGPU] Module {module_path} not found - skipping")
        return False
    logger.debug(f"[MultiGPU] Found {module_path}, creating compatible MultiGPU nodes")
    return True

current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()

def set_current_device(device):
    """Set the current device context for MultiGPU operations."""
    global current_device
    current_device = device
    logger.debug(f"[MultiGPU Initialization] current_device set to: {device}")

def set_current_text_encoder_device(device):
    """Set the current text encoder device context for CLIP models."""
    global current_text_encoder_device
    current_text_encoder_device = device
    logger.debug(f"[MultiGPU Initialization] current_text_encoder_device set to: {device}")

def get_torch_device_patched():
    """Return MultiGPU-aware device selection for patched mm.get_torch_device."""
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    logger.debug(f"[MultiGPU Core Patching] get_torch_device_patched returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    """Return MultiGPU-aware text encoder device for patched mm.text_encoder_device."""
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    logger.info(f"[MultiGPU Core Patching] text_encoder_device_patched returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device

logger.info(f"[MultiGPU Core Patching] Patching mm.get_torch_device and mm.text_encoder_device")
logger.info(f"[MultiGPU DEBUG] Initial current_device: {current_device}")
logger.info(f"[MultiGPU DEBUG] Initial current_text_encoder_device: {current_text_encoder_device}")
mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched

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
    UNetLoaderLP,
)

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

from .wrappers import (
    override_class,
    override_class_clip,
    override_class_clip_no_device,
    override_class_with_distorch_gguf,
    override_class_with_distorch_gguf_v2,
    override_class_with_distorch_clip,
    override_class_with_distorch_clip_no_device,
    override_class_with_distorch,
    override_class_with_distorch_safetensor_v2,
    override_class_with_distorch_safetensor_v2_clip,
    override_class_with_distorch_safetensor_v2_clip_no_device,
)
from .distorch_2 import (
    safetensor_allocation_store,
    create_safetensor_model_hash,
    register_patched_safetensor_modelpatcher,
    analyze_safetensor_loading,
    calculate_safetensor_vvram_allocation,
)

from .checkpoint_multigpu import (
    CheckpointLoaderAdvancedMultiGPU,
    CheckpointLoaderAdvancedDisTorch2MultiGPU
)

NODE_CLASS_MAPPINGS = {
    "DeviceSelectorMultiGPU": DeviceSelectorMultiGPU,
    "HunyuanVideoEmbeddingsAdapter": HunyuanVideoEmbeddingsAdapter,
    "CheckpointLoaderAdvancedMultiGPU": CheckpointLoaderAdvancedMultiGPU,
    "CheckpointLoaderAdvancedDisTorch2MultiGPU": CheckpointLoaderAdvancedDisTorch2MultiGPU,
    "UNetLoaderLP": UNetLoaderLP,
}

NODE_CLASS_MAPPINGS["UNETLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
NODE_CLASS_MAPPINGS["TripleCLIPLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
NODE_CLASS_MAPPINGS["DiffusersLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
NODE_CLASS_MAPPINGS["DiffControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])
NODE_CLASS_MAPPINGS["UNETLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
NODE_CLASS_MAPPINGS["TripleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
NODE_CLASS_MAPPINGS["DiffusersLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
NODE_CLASS_MAPPINGS["DiffControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

logger.info("[MultiGPU] Initiating custom_node Registration. . .")
dash_line = "-" * 47
fmt_reg = "{:<30}{:>5}{:>10}"
logger.info(dash_line)
logger.info(fmt_reg.format("custom_node", "Found", "Nodes"))
logger.info(dash_line)

registration_data = []

def register_and_count(module_names, node_map):
    """Register MultiGPU node wrappers for detected custom node modules."""
    found = False
    for name in module_names:
        if check_module_exists(name):
            found = True
            break
    
    count = 0
    if found:
        initial_len = len(NODE_CLASS_MAPPINGS)
        for key, value in node_map.items():
            NODE_CLASS_MAPPINGS[key] = value
        count = len(NODE_CLASS_MAPPINGS) - initial_len
        
    registration_data.append({"name": module_names[0], "found": "Y" if found else "N", "count": count})
    return found

ltx_nodes = {"LTXVLoaderMultiGPU": override_class(LTXVLoader)}
register_and_count(["ComfyUI-LTXVideo", "comfyui-ltxvideo"], ltx_nodes)

florence_nodes = {
    "Florence2ModelLoaderMultiGPU": override_class(Florence2ModelLoader),
    "DownloadAndLoadFlorence2ModelMultiGPU": override_class(DownloadAndLoadFlorence2Model)
}
register_and_count(["ComfyUI-Florence2", "comfyui-florence2"], florence_nodes)

nf4_nodes = {"CheckpointLoaderNF4MultiGPU": override_class(CheckpointLoaderNF4)}
register_and_count(["ComfyUI_bitsandbytes_NF4", "comfyui_bitsandbytes_nf4"], nf4_nodes)

flux_controlnet_nodes = {"LoadFluxControlNetMultiGPU": override_class(LoadFluxControlNet)}
register_and_count(["x-flux-comfyui"], flux_controlnet_nodes)

mmaudio_nodes = {
    "MMAudioModelLoaderMultiGPU": override_class(MMAudioModelLoader),
    "MMAudioFeatureUtilsLoaderMultiGPU": override_class(MMAudioFeatureUtilsLoader),
    "MMAudioSamplerMultiGPU": override_class(MMAudioSampler)
}
register_and_count(["ComfyUI-MMAudio", "comfyui-mmaudio"], mmaudio_nodes)

gguf_nodes = {
    "UnetLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_gguf(UnetLoaderGGUF),
    "UnetLoaderGGUFAdvancedDisTorchMultiGPU": override_class_with_distorch_gguf(UnetLoaderGGUFAdvanced),
    "CLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip(CLIPLoaderGGUF),
    "DualCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip(DualCLIPLoaderGGUF),
    "TripleCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip_no_device(TripleCLIPLoaderGGUF),
    "QuadrupleCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip_no_device(QuadrupleCLIPLoaderGGUF),
    "UnetLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2(UnetLoaderGGUF),
    "UnetLoaderGGUFAdvancedDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2(UnetLoaderGGUFAdvanced),
    "CLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip(CLIPLoaderGGUF),
    "DualCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip(DualCLIPLoaderGGUF),
    "TripleCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip_no_device(TripleCLIPLoaderGGUF),
    "QuadrupleCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip_no_device(QuadrupleCLIPLoaderGGUF),
    "UnetLoaderGGUFMultiGPU": override_class(UnetLoaderGGUF),
    "UnetLoaderGGUFAdvancedMultiGPU": override_class(UnetLoaderGGUFAdvanced),
    "CLIPLoaderGGUFMultiGPU": override_class_clip(CLIPLoaderGGUF),
    "DualCLIPLoaderGGUFMultiGPU": override_class_clip(DualCLIPLoaderGGUF),
    "TripleCLIPLoaderGGUFMultiGPU": override_class_clip_no_device(TripleCLIPLoaderGGUF),
    "QuadrupleCLIPLoaderGGUFMultiGPU": override_class_clip_no_device(QuadrupleCLIPLoaderGGUF)
}
register_and_count(["ComfyUI-GGUF", "comfyui-gguf"], gguf_nodes)

pulid_nodes = {
    "PulidModelLoaderMultiGPU": override_class(PulidModelLoader),
    "PulidInsightFaceLoaderMultiGPU": override_class(PulidInsightFaceLoader),
    "PulidEvaClipLoaderMultiGPU": override_class(PulidEvaClipLoader)
}
register_and_count(["PuLID_ComfyUI", "pulid_comfyui"], pulid_nodes)

hunyuan_nodes = {
    "HyVideoModelLoaderMultiGPU": override_class(HyVideoModelLoader),
    "HyVideoVAELoaderMultiGPU": override_class(HyVideoVAELoader),
    "DownloadAndLoadHyVideoTextEncoderMultiGPU": override_class(DownloadAndLoadHyVideoTextEncoder)
}
register_and_count(["ComfyUI-HunyuanVideoWrapper", "comfyui-hunyuanvideowrapper"], hunyuan_nodes)

wanvideo_nodes = {
    "WanVideoModelLoaderMultiGPU": WanVideoModelLoader,
    "WanVideoModelLoaderMultiGPU_2": WanVideoModelLoader_2,
    "WanVideoVAELoaderMultiGPU": WanVideoVAELoader,
    "LoadWanVideoT5TextEncoderMultiGPU": LoadWanVideoT5TextEncoder,
    "LoadWanVideoClipTextEncoderMultiGPU": LoadWanVideoClipTextEncoder,
    "WanVideoTextEncodeMultiGPU": WanVideoTextEncode,
    "WanVideoBlockSwapMultiGPU": WanVideoBlockSwap,
    "WanVideoSamplerMultiGPU": WanVideoSampler
}
register_and_count(["ComfyUI-WanVideoWrapper", "comfyui-wanvideowrapper"], wanvideo_nodes)

for item in registration_data:
    logger.info(fmt_reg.format(item['name'], item['found'], str(item['count'])))
logger.info(dash_line)

logger.info(f"[MultiGPU] Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")