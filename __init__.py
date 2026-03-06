import torch
import logging
import weakref
import os
import copy
import json
import importlib
from datetime import datetime
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

FOCUS_LOG_LEVEL = logging.INFO + 5
logging.addLevelName(FOCUS_LOG_LEVEL, "FOCUS")

if not hasattr(logging.Logger, "focus"):
    def focus(self, message, *args, **kwargs):
        if self.isEnabledFor(FOCUS_LOG_LEVEL):
            self._log(FOCUS_LOG_LEVEL, message, args, **kwargs)

    logging.Logger.focus = focus  # type: ignore[attr-defined]

if not logger.handlers:
    log_level = logging.DEBUG if DEBUG_LOG else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    json_log_path = os.environ.get("MGPU_JSON_LOG_PATH")
    json_static_fields = {}
    if json_log_path:
        try:
            json_static_fields = json.loads(os.environ.get("MGPU_JSON_STATIC_FIELDS", "{}"))
        except json.JSONDecodeError:
            json_static_fields = {}

        level_aliases = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "FOCUS": FOCUS_LOG_LEVEL,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }

        json_min_level = FOCUS_LOG_LEVEL
        configured_min_level = os.environ.get("MGPU_JSON_MIN_LEVEL")
        if configured_min_level:
            value = configured_min_level.strip()
            upper_value = value.upper()
            if upper_value in level_aliases:
                json_min_level = level_aliases[upper_value]
            else:
                try:
                    json_min_level = int(value)
                except ValueError:
                    json_min_level = FOCUS_LOG_LEVEL

        class JsonLineFileHandler(logging.Handler):
            def __init__(self, path, static_fields, min_level, overwrite):
                super().__init__()
                self.path = Path(path)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.static_fields = static_fields
                self.setLevel(min_level)
                if overwrite:
                    try:
                        with self.path.open("w", encoding="utf-8") as handle:
                            handle.write("")
                    except OSError:
                        pass

            def emit(self, record):
                message = record.getMessage()
                category = None
                if message.startswith("[") and "]" in message:
                    bracket_split = message.split("]", 1)
                    category = bracket_split[0].strip("[]")
                payload = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "name": record.name,
                    "message": message,
                }
                if category:
                    payload["event_category"] = category
                if hasattr(record, "mgpu_context") and isinstance(record.mgpu_context, dict):
                    payload.update(record.mgpu_context)
                workflow_id = os.environ.get("MGPU_JSON_WORKFLOW")
                prompt_id = os.environ.get("MGPU_JSON_PROMPT")
                if workflow_id:
                    payload.setdefault("workflow_id", workflow_id)
                if prompt_id:
                    payload.setdefault("prompt_id", prompt_id)
                if self.static_fields:
                    payload.update(self.static_fields)
                try:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
                except OSError:
                    # Fail silently for JSON logging so primary logging continues.
                    pass

        overwrite_value = os.environ.get("MGPU_JSON_OVERWRITE", "true").strip().lower()
        overwrite_enabled = overwrite_value not in {"0", "false", "no"}

        logger.addHandler(JsonLineFileHandler(json_log_path, json_static_fields, json_min_level, overwrite_enabled))

def mgpu_mm_log_method(self, msg):
    """Add MultiGPU model management logging method to logger instance."""
    if MGPU_MM_LOG:
        self.focus(
            f"[MultiGPU Model Management] {msg}",
            extra={"mgpu_context": {"component": "model_management"}},
        )
logger.mgpu_mm_log = mgpu_mm_log_method.__get__(logger, type(logger))

def _normalize_module_name(module_name):
    """Normalize a custom node directory name for tolerant matching."""
    return "".join(char for char in os.path.basename(module_name).lower() if char.isalnum())

def check_module_exists(module_path):
    """Check if a custom node module exists in ComfyUI custom_nodes directory."""
    custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")
    normalized_module_path = _normalize_module_name(module_path)

    for custom_nodes_path in custom_nodes_paths:
        full_path = os.path.join(custom_nodes_path, module_path)
        logger.debug(f"[MultiGPU] Checking for module at {full_path}")
        if os.path.isdir(full_path):
            logger.debug(f"[MultiGPU] Found exact module match for {module_path} at {full_path}")
            return True

    for custom_nodes_path in custom_nodes_paths:
        try:
            with os.scandir(custom_nodes_path) as entries:
                for entry in entries:
                    if not entry.is_dir():
                        continue
                    if _normalize_module_name(entry.name) == normalized_module_path:
                        logger.debug(f"[MultiGPU] Found normalized module match for {module_path} at {entry.path}")
                        return True
        except OSError:
            continue

    logger.debug(f"[MultiGPU] Module {module_path} not found - skipping")
    return False

current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()
current_unet_offload_device = mm.unet_offload_device()

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

def set_current_unet_offload_device(device):
    """Set the current UNet offload device context."""
    global current_unet_offload_device
    current_unet_offload_device = device
    logger.debug(f"[MultiGPU Initialization] current_unet_offload_device set to: {device}")


def get_current_device():
    """Get the current device context for MultiGPU operations at runtime."""
    return current_device


def get_current_text_encoder_device():
    """Get the current text encoder device context for CLIP models at runtime."""
    return current_text_encoder_device


def get_current_unet_offload_device():
    """Get the current UNet offload device context at runtime."""
    return current_unet_offload_device

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

def unet_offload_device_patched():
    """Return MultiGPU-aware UNet offload device for patched mm.unet_offload_device."""
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_unet_offload_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_unet_offload_device) if str(current_unet_offload_device) in devs else torch.device("cpu")
    logger.debug(f"[MultiGPU Core Patching] unet_offload_device_patched returning device: {device} (current_unet_offload_device={current_unet_offload_device})")
    return device

def _patch_comfy_kitchen_dlpack_device_guard():
    """Guard comfy_kitchen DLPack export by switching to the tensor's CUDA device."""
    try:
        comfy_kitchen_cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    except ImportError:
        logger.debug("[MultiGPU] comfy_kitchen not found - skipping CUDA DLPack compat patch")
        return False

    wrap_for_dlpack = getattr(comfy_kitchen_cuda, "_wrap_for_dlpack", None)
    if wrap_for_dlpack is None:
        logger.debug("[MultiGPU] comfy_kitchen.backends.cuda._wrap_for_dlpack not found - skipping compat patch")
        return False

    if getattr(wrap_for_dlpack, "_multigpu_cuda_device_guard", False):
        return True

    def wrap_for_dlpack_with_device_guard(*args, **kwargs):
        tensor = args[0] if args else kwargs.get("tensor")
        previous_device_index = None
        switched_device = False

        if isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.device.index is not None:
            previous_device_index = torch.cuda.current_device()
            if previous_device_index != tensor.device.index:
                torch.cuda.set_device(tensor.device.index)
                switched_device = True

        try:
            return wrap_for_dlpack(*args, **kwargs)
        finally:
            if switched_device and previous_device_index is not None:
                torch.cuda.set_device(previous_device_index)

    wrap_for_dlpack_with_device_guard._multigpu_cuda_device_guard = True
    comfy_kitchen_cuda._wrap_for_dlpack = wrap_for_dlpack_with_device_guard
    logger.info("[MultiGPU] Applied comfy_kitchen CUDA DLPack device guard patch")
    return True

logger.info(f"[MultiGPU Core Patching] Patching mm.get_torch_device, mm.text_encoder_device, mm.unet_offload_device")
logger.info(f"[MultiGPU DEBUG] Initial current_device: {current_device}")
logger.info(f"[MultiGPU DEBUG] Initial current_text_encoder_device: {current_text_encoder_device}")
logger.info(f"[MultiGPU DEBUG] Initial current_unet_offload_device: {current_unet_offload_device}")

mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched
mm.unet_offload_device = unet_offload_device_patched
_patch_comfy_kitchen_dlpack_device_guard()

from .nodes import (
    DeviceSelectorMultiGPU,
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
    UNetLoaderLP,
)

from .wrappers import (
    override_class,
    override_class_offload,
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
    register_patched_safetensor_modelpatcher,
    analyze_safetensor_loading,
    calculate_safetensor_vvram_allocation,
)

from .checkpoint_multigpu import (
    CheckpointLoaderAdvancedMultiGPU,
    CheckpointLoaderAdvancedDisTorch2MultiGPU
)

def _load_wanvideo_nodes():
    from .wanvideo import (
        LoadWanVideoT5TextEncoder,
        WanVideoTextEncode,
        WanVideoTextEncodeCached,
        WanVideoTextEncodeSingle,
        WanVideoVAELoader,
        WanVideoTinyVAELoader,
        WanVideoBlockSwap,
        WanVideoImageToVideoEncode,
        WanVideoDecode,
        WanVideoModelLoader,
        WanVideoSampler,
        WanVideoVACEEncode,
        WanVideoEncode,
        LoadWanVideoClipTextEncoder,
        WanVideoClipVisionEncode,
        WanVideoControlnetLoader,
        FantasyTalkingModelLoader,
        Wav2VecModelLoader,
        WanVideoUni3C_ControlnetLoader,
        DownloadAndLoadWav2VecModel,
    )

    return {
        "LoadWanVideoT5TextEncoderMultiGPU": LoadWanVideoT5TextEncoder,
        "WanVideoTextEncodeMultiGPU": WanVideoTextEncode,
        "WanVideoTextEncodeCachedMultiGPU": WanVideoTextEncodeCached,
        "WanVideoTextEncodeSingleMultiGPU": WanVideoTextEncodeSingle,
        "WanVideoVAELoaderMultiGPU": WanVideoVAELoader,
        "WanVideoTinyVAELoaderMultiGPU": WanVideoTinyVAELoader,
        "WanVideoBlockSwapMultiGPU": WanVideoBlockSwap,
        "WanVideoImageToVideoEncodeMultiGPU": WanVideoImageToVideoEncode,
        "WanVideoDecodeMultiGPU": WanVideoDecode,
        "WanVideoModelLoaderMultiGPU": WanVideoModelLoader,
        "WanVideoSamplerMultiGPU": WanVideoSampler,
        "WanVideoVACEEncodeMultiGPU": WanVideoVACEEncode,
        "WanVideoEncodeMultiGPU": WanVideoEncode,
        "LoadWanVideoClipTextEncoderMultiGPU": LoadWanVideoClipTextEncoder,
        "WanVideoClipVisionEncodeMultiGPU": WanVideoClipVisionEncode,
        "WanVideoControlnetLoaderMultiGPU": WanVideoControlnetLoader,
        "FantasyTalkingModelLoaderMultiGPU": FantasyTalkingModelLoader,
        "Wav2VecModelLoaderMultiGPU": Wav2VecModelLoader,
        "WanVideoUni3C_ControlnetLoaderMultiGPU": WanVideoUni3C_ControlnetLoader,
        "DownloadAndLoadWav2VecModelMultiGPU": DownloadAndLoadWav2VecModel,
    }

NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderAdvancedMultiGPU": CheckpointLoaderAdvancedMultiGPU,
    "CheckpointLoaderAdvancedDisTorch2MultiGPU": CheckpointLoaderAdvancedDisTorch2MultiGPU,
    "DeviceSelectorMultiGPU": DeviceSelectorMultiGPU,
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
        try:
            resolved_node_map = node_map() if callable(node_map) else node_map
        except Exception as exc:
            logger.warning(f"[MultiGPU] Failed to register nodes for {module_names[0]}: {exc}")
            resolved_node_map = {}

        initial_len = len(NODE_CLASS_MAPPINGS)
        for key, value in resolved_node_map.items():
            NODE_CLASS_MAPPINGS[key] = value
        count = len(NODE_CLASS_MAPPINGS) - initial_len
        
    registration_data.append({"name": module_names[0], "found": "Y" if found else "N", "count": count})
    return found

ltx_nodes = {"LTXVLoaderMultiGPU": override_class(LTXVLoader)}
register_and_count(["ComfyUI-LTXVideo", "comfyui-ltxvideo"], ltx_nodes)

florence_nodes = {
    "Florence2ModelLoaderMultiGPU": override_class_offload(Florence2ModelLoader),
    "DownloadAndLoadFlorence2ModelMultiGPU": override_class_offload(DownloadAndLoadFlorence2Model)
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

register_and_count(["ComfyUI-WanVideoWrapper", "comfyui-wanvideowrapper"], _load_wanvideo_nodes)

for item in registration_data:
    logger.info(fmt_reg.format(item['name'], item['found'], str(item['count'])))
logger.info(dash_line)

logger.info(f"[MultiGPU] Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
