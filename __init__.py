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


MGPU_MM_LOG = True 

# Set to "E" for Engineering (DEBUG) or "P" for Production (INFO)
LOG_LEVEL = "P"
logger = logging.getLogger("MultiGPU")
logger.propagate = False

if not logger.handlers:
    log_level = logging.DEBUG if LOG_LEVEL == "E" else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

def mgpu_mm_log_method(self, msg):
    if MGPU_MM_LOG:
        self.info(f"[MultiGPU Model Management] {msg}")
logger.mgpu_mm_log = mgpu_mm_log_method.__get__(logger, type(logger))

# Global device state management
current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()

def set_current_device(device):
    global current_device
    current_device = device
    logger.debug(f"[MultiGPU Initialization] current_device set to: {device}")

def set_current_text_encoder_device(device):
    global current_text_encoder_device
    current_text_encoder_device = device
    logger.debug(f"[MultiGPU Initialization] current_text_encoder_device set to: {device}")

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

            if device is not None:
                set_current_device(device)
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
            if device is not None:
                set_current_text_encoder_device(device)
            kwargs['device'] = 'default'
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            return out

    return NodeOverride

def override_class_clip_no_device(cls):
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
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    logger.debug(f"[MultiGPU Core Patching] get_torch_device_patched returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    logger.debug(f"[MultiGPU Core Patching] text_encoder_device_patched returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device


logger.info(f"[MultiGPU Core Patching] Patching mm.get_torch_device and mm.text_encoder_device")
logger.debug(f"[MultiGPU DEBUG] Initial current_device: {current_device}")
logger.debug(f"[MultiGPU DEBUG] Initial current_text_encoder_device: {current_text_encoder_device}")
mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched

def check_module_exists(module_path):
    full_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], module_path)
    logger.debug(f"[MultiGPU] Checking for module at {full_path}")
    if not os.path.exists(full_path):
        logger.debug(f"[MultiGPU] Module {module_path} not found - skipping")
        return False
    logger.debug(f"[MultiGPU] Found {module_path}, creating compatible MultiGPU nodes")
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
    UNetLoaderLP,
    FullCleanupMultiGPU,
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
    override_class_with_distorch_clip_no_device,
    override_class_with_distorch
)

# Import from distorch_2.py for DisTorch v2 SafeTensor support
from .distorch_2 import (
    safetensor_allocation_store,
    create_safetensor_model_hash,
    register_patched_safetensor_modelpatcher,
    analyze_safetensor_loading,
    calculate_safetensor_vvram_allocation,
    override_class_with_distorch_safetensor_v2,
    override_class_with_distorch_safetensor_v2_clip,
    override_class_with_distorch_safetensor_v2_clip_no_device
)

logger.info("[MultiGPU Core Patching] Patching mm.soft_empty_cache for Comprehensive Memory Management (VRAM + CPU + Store Pruning)")

original_soft_empty_cache = mm.soft_empty_cache

def soft_empty_cache_distorch2_patched(force=False):
    """
    Patched mm.soft_empty_cache.
    - Prunes DisTorch store bookkeeping to avoid stale references
    - Manages VRAM: if DisTorch2 models are active, clear allocator caches on all devices;
      otherwise delegate to original mm.soft_empty_cache.
    - Manages CPU RAM: adaptive threshold-based PromptExecutor cache reset;
      and force-triggered reset when explicitly requested (mirrors ComfyUI 'Free memory' button).
    """
    multigpu_memory_log("patched_soft_empty", f"start:force={force}")
    is_distorch_active = False

    # Detect DisTorch2-managed models
    logger.mgpu_mm_log(f"[DETECT_DEBUG] Checking DisTorch2 active status - loaded models: {len(mm.current_loaded_models)}, store entries: {len(safetensor_allocation_store)}")
    
    for i, lm in enumerate(mm.current_loaded_models):
        mp = lm.model  # weakref call to ModelPatcher
        if mp is not None:
            try:
                model_hash = create_safetensor_model_hash(mp, "cache_patch_check")
                in_store = model_hash in safetensor_allocation_store
                alloc_value = safetensor_allocation_store.get(model_hash, "")
                model_name = type(getattr(mp, 'model', mp)).__name__
                unload_distorch_model = getattr(getattr(mp, 'model', None), '_mgpu_unload_distorch_model', False)
                
                logger.mgpu_mm_log(f"[DETECT_DEBUG] Model {i}: {model_name}, hash={model_hash[:8]}, in_store={in_store}, alloc_value='{alloc_value}', unload_distorch_model={unload_distorch_model}")
                
                if in_store and alloc_value:
                    is_distorch_active = True
                    logger.mgpu_mm_log(f"[DETECT_DEBUG] DisTorch2 ACTIVE detected on model: {model_name}")
                    break
            except Exception as e:
                logger.mgpu_mm_log(f"[DETECT_DEBUG] Model {i}: Error during detection - {e}")
    
    logger.mgpu_mm_log(f"[DETECT_DEBUG] Final DisTorch2 active status: {is_distorch_active}")

    # Phase 2: adaptive CPU memory management
    check_cpu_memory_threshold()

    # VRAM allocator management
    if is_distorch_active:
        logger.mgpu_mm_log("DisTorch2 active: clearing allocator caches on all devices (VRAM)")
        soft_empty_cache_multigpu()
    else:
        logger.mgpu_mm_log("DisTorch2 not active: delegating allocator cache clear (VRAM) to original mm.soft_empty_cache")
        original_soft_empty_cache(force)
        # Optional: return CPU heap to OS (not part of Comfy Core)

    # Phase 1/3: forced executor reset mirrors ComfyUI 'Free memory' semantics
    if force:
        logger.mgpu_mm_log("Force flag active: triggering executor cache reset (CPU)")
        trigger_executor_cache_reset(reason="forced_soft_empty", force=True)
    multigpu_memory_log("patched_soft_empty", "end")

mm.soft_empty_cache = soft_empty_cache_distorch2_patched

LARGE_MODEL_THRESHOLD = 2 * (1024**3)  # 2 GB threshold for "large" models

# Import advanced checkpoint loaders
from .checkpoint_multigpu import (
    CheckpointLoaderAdvancedMultiGPU,
    CheckpointLoaderAdvancedDisTorch2MultiGPU
)

# Initialize NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "DeviceSelectorMultiGPU": DeviceSelectorMultiGPU,
    "HunyuanVideoEmbeddingsAdapter": HunyuanVideoEmbeddingsAdapter,
    "CheckpointLoaderAdvancedMultiGPU": CheckpointLoaderAdvancedMultiGPU,
    "CheckpointLoaderAdvancedDisTorch2MultiGPU": CheckpointLoaderAdvancedDisTorch2MultiGPU,
    "UNetLoaderLP": UNetLoaderLP,
}

# Standard MultiGPU nodes
NODE_CLASS_MAPPINGS["UNETLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
if "QuadrupleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
if "DiffusersLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffusersLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
if "DiffControlNetLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

# DisTorch 2 SafeTensor nodes for FLUX and other safetensor models
NODE_CLASS_MAPPINGS["UNETLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["VAELoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["CLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["DualCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["TripleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
if "QuadrupleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["CLIPVisionLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["ControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
if "DiffusersLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffusersLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
if "DiffControlNetLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["DiffControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

# --- Registration Table ---
logger.info("[MultiGPU] Initiating custom_node Registration. . .")
dash_line = "-" * 47
fmt_reg = "{:<30}{:>5}{:>10}"
logger.info(dash_line)
logger.info(fmt_reg.format("custom_node", "Found", "Nodes"))
logger.info(dash_line)

registration_data = []

def register_and_count(module_names, node_map):
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

# ComfyUI-LTXVideo
ltx_nodes = {"LTXVLoaderMultiGPU": override_class(LTXVLoader)}
register_and_count(["ComfyUI-LTXVideo", "comfyui-ltxvideo"], ltx_nodes)

# ComfyUI-Florence2
florence_nodes = {
    "Florence2ModelLoaderMultiGPU": override_class(Florence2ModelLoader),
    "DownloadAndLoadFlorence2ModelMultiGPU": override_class(DownloadAndLoadFlorence2Model)
}
register_and_count(["ComfyUI-Florence2", "comfyui-florence2"], florence_nodes)

# ComfyUI_bitsandbytes_NF4
nf4_nodes = {"CheckpointLoaderNF4MultiGPU": override_class(CheckpointLoaderNF4)}
register_and_count(["ComfyUI_bitsandbytes_NF4", "comfyui_bitsandbytes_nf4"], nf4_nodes)

# x-flux-comfyui
flux_controlnet_nodes = {"LoadFluxControlNetMultiGPU": override_class(LoadFluxControlNet)}
register_and_count(["x-flux-comfyui"], flux_controlnet_nodes)

# ComfyUI-MMAudio
mmaudio_nodes = {
    "MMAudioModelLoaderMultiGPU": override_class(MMAudioModelLoader),
    "MMAudioFeatureUtilsLoaderMultiGPU": override_class(MMAudioFeatureUtilsLoader),
    "MMAudioSamplerMultiGPU": override_class(MMAudioSampler)
}
register_and_count(["ComfyUI-MMAudio", "comfyui-mmaudio"], mmaudio_nodes)

# ComfyUI-GGUF
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

# PuLID_ComfyUI
pulid_nodes = {
    "PulidModelLoaderMultiGPU": override_class(PulidModelLoader),
    "PulidInsightFaceLoaderMultiGPU": override_class(PulidInsightFaceLoader),
    "PulidEvaClipLoaderMultiGPU": override_class(PulidEvaClipLoader)
}
register_and_count(["PuLID_ComfyUI", "pulid_comfyui"], pulid_nodes)

# ComfyUI-HunyuanVideoWrapper
hunyuan_nodes = {
    "HyVideoModelLoaderMultiGPU": override_class(HyVideoModelLoader),
    "HyVideoVAELoaderMultiGPU": override_class(HyVideoVAELoader),
    "DownloadAndLoadHyVideoTextEncoderMultiGPU": override_class(DownloadAndLoadHyVideoTextEncoder)
}
register_and_count(["ComfyUI-HunyuanVideoWrapper", "comfyui-hunyuanvideowrapper"], hunyuan_nodes)

# ComfyUI-WanVideoWrapper
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

# Print the registration table
for item in registration_data:
    logger.info(fmt_reg.format(item['name'], item['found'], str(item['count'])))
logger.info(dash_line)


# Register maintenance node
NODE_CLASS_MAPPINGS["FullCleanupMultiGPU"] = FullCleanupMultiGPU

logger.info(f"[MultiGPU] Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
