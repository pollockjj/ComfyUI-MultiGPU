import torch
import logging
import os
import copy
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
from .device_utils import get_device_list, is_accelerator_available, soft_empty_cache_multigpu

# --- DisTorch V2 Logging Configuration ---
# Set to "E" for Engineering (DEBUG) or "P" for Production (INFO)
LOG_LEVEL = "P"

# Configure logger
logger = logging.getLogger("MultiGPU")
logger.propagate = False

if not logger.handlers:
    log_level = logging.DEBUG if LOG_LEVEL == "E" else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

MEMORY_LOG = True

def memory_method(self, msg):
    if MEMORY_LOG:
        self.info(msg)
logger.memory = memory_method.__get__(logger, type(logger))


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

logger.info("[MultiGPU Core Patching] Patching mm.soft_empty_cache for DisTorch2 Multi-Device Allocation/Clearing")

original_soft_empty_cache = mm.soft_empty_cache

def soft_empty_cache_distorch2_patched(force=False):
    """
    Patched mm.soft_empty_cache. If DisTorch2 models are active, clear cache on ALL devices.
    Otherwise, execute original ComfyUI behavior.
    """
    is_distorch_active = False

    # Check if any loaded model is managed by DisTorch2 using the allocation store
    for lm in mm.current_loaded_models:
        mp = lm.model  # weakref call to ModelPatcher
        if mp is not None:
            model_hash = create_safetensor_model_hash(mp, "cache_patch_check")
            if model_hash in safetensor_allocation_store and safetensor_allocation_store[model_hash]:
                is_distorch_active = True
                break

    if is_distorch_active:
        logger.info("[MultiGPU Core Patching] DisTorch2 active: clearing caches on all devices")
        soft_empty_cache_multigpu()
    else:
        logger.info("[MultiGPU Core Patching] DisTorch2 not active: delegating to original mm.soft_empty_cache")
        original_soft_empty_cache(force)

mm.soft_empty_cache = soft_empty_cache_distorch2_patched

LARGE_MODEL_THRESHOLD = 2 * (1024**3)  # 2 GB threshold for "large" models

# Patch only once (handles reloads)
if hasattr(mm, 'load_models_gpu') and not hasattr(mm.load_models_gpu, "_distorch2_proactive_patched"):
    logger.info("[MultiGPU Core Patching] Patching mm.load_models_gpu for DisTorch2 proactive unloading")

    original_load_models_gpu = mm.load_models_gpu

    def patched_load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
        """
        Proactively unload large models that are not needed when loading a large DisTorch2 model.
        This frees both compute and donor device memory ahead of ComfyUI's compute-only check.
        """
        # Validate models argument loudly
        if not isinstance(models, (list, tuple, set)):
            logger.error("[MultiGPU Core Patching] CRITICAL: mm.load_models_gpu 'models' is not a list/tuple/set. Bypassing proactive patch.")
            return original_load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

        # Detect incoming large DisTorch2 request
        incoming_is_distorch = False
        incoming_distorch_nonzero = False
        incoming_is_large = False
        incoming_patchers = set()
        incoming_loaded_names = []
        incoming_allowed_devices = None

        for lm in models:
            # Identify ModelPatcher (prefer direct; fall back to .patcher)
            if hasattr(lm, "load_device"):
                patcher = lm
            elif hasattr(lm, "patcher"):
                patcher = lm.patcher
            else:
                patcher = None

            model_for_hash = patcher if patcher is not None else getattr(lm, "model", lm)

            if patcher is not None:
                incoming_patchers.add(patcher)

                # Determine required memory directly from ModelPatcher (no wrapper; no side effects)
                device_str = str(patcher.load_device)
                if patcher.current_loaded_device() == patcher.load_device:
                    required_bytes = patcher.model_size() - patcher.loaded_size()
                else:
                    required_bytes = patcher.model_size()

                if required_bytes > LARGE_MODEL_THRESHOLD:
                    incoming_is_large = True
            else:
                device_str = "n/a"
                required_bytes = 0

            # Check DisTorch2 management via allocation store (unchanged trigger)
            model_hash = create_safetensor_model_hash(model_for_hash, "load_patch_check")
            if model_hash in safetensor_allocation_store and safetensor_allocation_store.get(model_hash):
                incoming_is_distorch = True
                if required_bytes > 0:
                    incoming_distorch_nonzero = True
                    if incoming_allowed_devices is None:
                        # Derive compute/donor devices from allocation string
                        alloc_str = safetensor_allocation_store.get(model_hash, "")
                        allowed = set()
                        if alloc_str:
                            parts = alloc_str.split("#", 1)
                            if len(parts) == 2 and parts[1]:
                                vram = parts[1]
                                segs = vram.split(";")
                                # compute device
                                if len(segs) >= 1 and segs[0]:
                                    allowed.add(segs[0].strip())
                                # donors list (comma-separated)
                                if len(segs) >= 3 and segs[2]:
                                    for d in segs[2].split(","):
                                        d = d.strip()
                                        if d:
                                            allowed.add(d)
                            else:
                                # Expert fraction string: "dev,fraction;dev2,fraction2;..."
                                for token in alloc_str.split(";"):
                                    if "," in token:
                                        dev, frac = token.split(",", 1)
                                        fs = frac.strip()
                                        numlike = fs.replace(".", "", 1).isdigit()
                                        if numlike and float(fs) > 0.0:
                                            allowed.add(dev.strip())
                        if not allowed:
                            allowed = {str(patcher.load_device), "cpu"}
                        incoming_allowed_devices = allowed

            # Log informational context with required bytes and device
            try:
                model_name = type(getattr(model_for_hash, "model", model_for_hash)).__name__
            except Exception:
                model_name = "UnknownModel"
            incoming_loaded_names.append(f"{model_name}:{required_bytes/(1024**3):.2f}GB req on {device_str}")

        logger.info(f"[MultiGPU Core Patching] load_models_gpu incoming set: large={incoming_is_large} distorch2={incoming_is_distorch} count={len(incoming_patchers)}")
        if incoming_loaded_names:
            logger.info(f"[MultiGPU Core Patching] Incoming models summary: {', '.join(incoming_loaded_names)}")

        if incoming_distorch_nonzero:
            logger.info("[MultiGPU Core Patching] Non-Zero incoming DisTorch2 model detected. Initiating proactive unload.")
            if not hasattr(mm, 'current_loaded_models'):
                raise AttributeError("comfy.model_management is missing 'current_loaded_models'. Proactive unload check failed.")

            to_unload_indices = []
            unload_summaries = []
            needed_patchers = incoming_patchers
            logger.info("[MultiGPU Core Patching] Incoming large DisTorch2 model detected. Initiating proactive unload of other large models.")

            # Iterate backwards to safely pop from list
            for i in range(len(mm.current_loaded_models) - 1, -1, -1):
                lm_cur = mm.current_loaded_models[i]
                mp_cur = getattr(lm_cur, 'model', None)
                if mp_cur is None:
                    continue  # already dead or cleaned up

                # Skip models needed for this load call
                if mp_cur in needed_patchers:
                    continue

                # Only consider models on compute/donor devices for this DisTorch2 load
                if incoming_allowed_devices is not None:
                    cur_dev_str = str(getattr(lm_cur, "device", ""))
                    if cur_dev_str not in incoming_allowed_devices:
                        continue

                # Determine size (prefer LoadedModel.model_memory)
                size_cur = 0
                if hasattr(lm_cur, 'model_memory'):
                    try:
                        size_cur = lm_cur.model_memory()
                    except Exception:
                        size_cur = 0
                if size_cur <= 0 and hasattr(mp_cur, 'model_size'):
                    size_cur = mp_cur.model_size()

                model_name = type(getattr(mp_cur, 'model', mp_cur)).__name__
                logger.info(f"[MultiGPU Core Patching] Unloading large model: {model_name} (~{size_cur/(1024**3):.2f}GB)")
                # Attempt full unload; unpatch_weights=True to release distributed allocations
                success = False
                if hasattr(lm_cur, 'model_unload'):
                    success = lm_cur.model_unload(memory_to_free=None, unpatch_weights=True)
                if success:
                    to_unload_indices.append(i)
                    unload_summaries.append(f"{model_name}:{size_cur/(1024**3):.2f}GB")
                else:
                    logger.warning(f"[MultiGPU Core Patching] Failed to fully unload model {model_name} (~{size_cur/(1024**3):.2f}GB)")

            # Remove from management list and clear caches
            unloaded_count = 0
            for idx in to_unload_indices:  # already in reverse order
                mm.current_loaded_models.pop(idx)
                unloaded_count += 1

            if unloaded_count > 0:
                logger.info(f"[MultiGPU Core Patching] Proactively unloaded {unloaded_count} large model(s): {', '.join(unload_summaries)}")
                logger.info("[MultiGPU Core Patching] Performing multi-device cache clear after proactive unload")
                # Force multi-device cache clear via patched soft_empty_cache (which detects DisTorch2)
                mm.soft_empty_cache(force=True)
            else:
                # Lineage-aligned cache clear when no unloads happened: apply core 25% rule, per DisTorch devices
                if incoming_allowed_devices is not None and mm.vram_state != mm.VRAMState.HIGH_VRAM:
                    triggered = []
                    for dev_str in incoming_allowed_devices:
                        try:
                            dev_obj = torch.device(dev_str)
                        except Exception:
                            continue
                        free_total, free_torch = mm.get_free_memory(dev_obj, torch_free_too=True)
                        # free_total: system free; free_torch: torch reserved-but-free
                        if free_torch > free_total * 0.25:
                            triggered.append(dev_str)
                    if triggered:
                        logger.info(f"[MultiGPU Core Patching] No unloads; 25% torch-cache rule triggered on: {', '.join(triggered)}. Calling soft_empty_cache()")
                        mm.soft_empty_cache(force=True)
                    else:
                        logger.info("[MultiGPU Core Patching] No unloads; 25% torch-cache rule not met on DisTorch devices; skipping cache clear")
                else:
                    logger.info("[MultiGPU Core Patching] No unload candidates matched criteria and either HIGH_VRAM or no DisTorch devices; skipping cache clear")
        elif incoming_is_distorch:
            logger.info("[MultiGPU Core Patching] Incoming DisTorch2 model requires 0.00GB; skipping proactive unload")

        # Continue with original behavior
        return original_load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

    # Mark and apply the patch
    patched_load_models_gpu._distorch2_proactive_patched = True
    mm.load_models_gpu = patched_load_models_gpu
else:
    if not hasattr(mm, 'load_models_gpu'):
        raise AttributeError("comfy.model_management is missing 'load_models_gpu'. Core patching failed.")
    else:
        logger.debug("[MultiGPU Core Patching] mm.load_models_gpu already patched; skipping")

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


logger.info(f"[MultiGPU] Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
