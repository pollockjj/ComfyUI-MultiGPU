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
    prune_distorch_stores,
    try_malloc_trim,
    track_modelpatcher,
    force_full_system_cleanup,
)

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

# --- MultiGPU Cleanup Policy Configuration ---
# Policy: off | threshold | every_load | every_load+threshold (alias threshold+every_load)
MGPU_CLEANUP_POLICY = os.getenv("MULTIGPU_CLEANUP_POLICY", "off").lower()
try:
    MGPU_CPU_RESET_THRESHOLD = float(os.getenv("MULTIGPU_CPU_RESET_THRESHOLD", "0.85"))
except Exception:
    MGPU_CPU_RESET_THRESHOLD = 0.85
# Malloc trim (not part of Comfy Core): on | off
MGPU_MALLOC_TRIM = os.getenv("MULTIGPU_MALLOC_TRIM", "on").lower()

logger.info(f"[MultiGPU Config] cleanup_policy={MGPU_CLEANUP_POLICY}, cpu_reset_threshold={MGPU_CPU_RESET_THRESHOLD:.2f}, malloc_trim={MGPU_MALLOC_TRIM}")

MGPU_MM_LOG = True

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


# ==========================================================================================
# Core Patching: ModelPatcher Lifecycle Tracking (__init__)
# ==========================================================================================
logger.info("[MultiGPU Core Patching] Applying ModelPatcher lifecycle tracking patch (__init__).")
if not hasattr(comfy.model_patcher.ModelPatcher, '_mgpu_lifecycle_patched'):
    try:
        _mgpu_original_modelpatcher_init = comfy.model_patcher.ModelPatcher.__init__

        def _mgpu_patched_modelpatcher_init(self, *args, **kwargs):
            _mgpu_original_modelpatcher_init(self, *args, **kwargs)
            # Track all ModelPatcher instances at construction time
            try:
                track_modelpatcher(self)
            except Exception:
                pass

        comfy.model_patcher.ModelPatcher.__init__ = _mgpu_patched_modelpatcher_init
        comfy.model_patcher.ModelPatcher._mgpu_lifecycle_patched = True
        logger.info("[MultiGPU Core Patching] ModelPatcher.__init__ patched for lifecycle tracking.")
    except Exception as e:
        logger.error(f"[MultiGPU Core Patching] FAILED to patch ModelPatcher.__init__: {e}")

# ==========================================================================================
# Core Patching: Fix Potential Reference Cycles in LoadedModel
# ==========================================================================================
if hasattr(mm, 'LoadedModel') and hasattr(mm.LoadedModel, '_set_model'):
    logger.info("[MultiGPU Core Patching] Patching mm.LoadedModel._set_model and _switch_parent to reduce reference cycles.")

    _mgpu_original_set_model = mm.LoadedModel._set_model

    def _mgpu_patched_set_model(self, model):
        patcher_id = id(model)
        # Ensure attributes exist
        if not hasattr(self, '_model'):
            self._model = None
        if not hasattr(self, '_parent_model'):
            self._parent_model = None
        if not hasattr(self, '_patcher_finalizer'):
            self._patcher_finalizer = None

        # Reset refs
        self._model = weakref.ref(model)
        self._parent_model = None

        # Detach any previous finalizer
        if self._patcher_finalizer is not None:
            try:
                self._patcher_finalizer.detach()
            except Exception:
                pass
            self._patcher_finalizer = None

        # If clone, set parent and attach a weakref-based finalizer
        parent = getattr(model, 'parent', None)
        if parent is not None:
            self._parent_model = weakref.ref(parent)
            self_weak = weakref.ref(self)

            def _mgpu_finalize_clone():
                s = self_weak()
                if s is not None and hasattr(s, '_switch_parent'):
                    logger.mgpu_mm_log(f"[MultiGPU_LoadedModel_Patch] Clone Patcher {patcher_id} GC'd. Switching LoadedModel to parent.")
                    s._switch_parent()
                else:
                    logger.mgpu_mm_log(f"[MultiGPU_LoadedModel_Patch] Clone Patcher {patcher_id} GC'd. LoadedModel already gone or missing _switch_parent.")

            try:
                self._patcher_finalizer = weakref.finalize(model, _mgpu_finalize_clone)
            except Exception:
                self._patcher_finalizer = None
        else:
            logger.mgpu_mm_log(f"[MultiGPU_LoadedModel_Patch] Set base model Patcher {patcher_id}.")

    mm.LoadedModel._set_model = _mgpu_patched_set_model

    # Patch _switch_parent to clear references explicitly
    if hasattr(mm.LoadedModel, '_switch_parent'):
        _mgpu_original_switch_parent = mm.LoadedModel._switch_parent

        def _mgpu_patched_switch_parent(self):
            _mgpu_original_switch_parent(self)
            # Clear parent and detach finalizer to avoid cycles
            if hasattr(self, '_parent_model'):
                self._parent_model = None
            if hasattr(self, '_patcher_finalizer') and self._patcher_finalizer is not None:
                try:
                    self._patcher_finalizer.detach()
                except Exception:
                    pass
                self._patcher_finalizer = None

        mm.LoadedModel._switch_parent = _mgpu_patched_switch_parent
    else:
        # Fallback if core ever changes
        def _mgpu_fallback_switch_parent(self):
            if hasattr(self, '_parent_model') and self._parent_model is not None:
                parent_model = self._parent_model()
                if parent_model is not None:
                    self._set_model(parent_model)
                self._parent_model = None
        mm.LoadedModel._switch_parent = _mgpu_fallback_switch_parent
else:
    logger.warning("[MultiGPU Core Patching] mm.LoadedModel not found or missing _set_model; skip cycle patch.")

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
    # Prune DisTorch stores before any clearing to drop stale references
    try:
        prune_distorch_stores()
    except Exception:
        pass
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
                keep_loaded = getattr(getattr(mp, 'model', None), '_mgpu_keep_loaded', False)
                
                logger.mgpu_mm_log(f"[DETECT_DEBUG] Model {i}: {model_name}, hash={model_hash[:8]}, in_store={in_store}, alloc_value='{alloc_value}', keep_loaded={keep_loaded}")
                
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
        if MGPU_MALLOC_TRIM != "off":
            try:
                try_malloc_trim()
            except Exception:
                pass

    # Phase 1/3: forced executor reset mirrors ComfyUI 'Free memory' semantics
    if force:
        logger.mgpu_mm_log("Force flag active: triggering executor cache reset (CPU)")
        trigger_executor_cache_reset(reason="forced_soft_empty", force=True)
    multigpu_memory_log("patched_soft_empty", "end")

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
        multigpu_memory_log("patched_load_models_gpu", "start")
        # Validate models argument loudly
        if not isinstance(models, (list, tuple, set)):
            logger.error("[MultiGPU Core Patching] CRITICAL: mm.load_models_gpu 'models' is not a list/tuple/set. Bypassing proactive patch.")
            return original_load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

        # Detect incoming DisTorch2 request
        incoming_is_distorch = False
        incoming_distorch_nonzero = False
        incoming_patchers = set()
        incoming_loaded_names = []
        incoming_allowed_devices = None
        incoming_compute_device = None
        incoming_required_bytes = 0
        incoming_compute_planned_bytes = 0

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
                        # Determine compute device and planned bytes from allocation string
                        alloc = safetensor_allocation_store.get(model_hash, "")
                        if "#" in alloc:
                            vram = alloc.split("#", 1)[1]
                            segs = vram.split(";")
                            if len(segs) >= 2 and segs[0]:
                                incoming_compute_device = segs[0].strip()
                                try:
                                    vvram_gb = float(segs[1])
                                    incoming_compute_planned_bytes = int(vvram_gb * (1024**3))
                                except Exception:
                                    incoming_compute_planned_bytes = 0
                        else:
                            # Expert fractions: "dev,fraction;dev2,fraction2;..."
                            tokens = [t for t in alloc.split(";") if "," in t]
                            frac_map = {}
                            for t in tokens:
                                dev, frac = t.split(",", 1)
                                try:
                                    frac_val = float(frac.strip())
                                except Exception:
                                    continue
                                frac_map[dev.strip()] = frac_val
                            if frac_map:
                                ld = str(patcher.load_device)
                                # Prefer the explicit load_device if present and > 0
                                target_dev = ld if (ld in frac_map and frac_map[ld] > 0.0) else None
                                if target_dev is None:
                                    # Otherwise pick highest positive fraction
                                    target_dev = max((d for d,v in frac_map.items() if v > 0.0), key=lambda d: frac_map[d], default=None)
                                if target_dev is not None:
                                    incoming_compute_device = target_dev
                                    total = mm.get_total_memory(torch.device(target_dev))
                                    incoming_compute_planned_bytes = int(frac_map[target_dev] * (total or 0))
                        if incoming_compute_device is None:
                            incoming_compute_device = str(patcher.load_device)
                        if incoming_compute_planned_bytes <= 0:
                            incoming_compute_planned_bytes = required_bytes

            # Log informational context with required bytes and device
            try:
                model_name = type(getattr(model_for_hash, "model", model_for_hash)).__name__
            except Exception:
                model_name = "UnknownModel"
            incoming_loaded_names.append(f"{model_name}:{required_bytes/(1024**3):.2f}GB req on {device_str}")

        if incoming_loaded_names:
            logger.mgpu_mm_log(f"Incoming models summary: {', '.join(incoming_loaded_names)}")

        if incoming_distorch_nonzero:
            logger.mgpu_mm_log("Non-Zero incoming DisTorch2 model detected. Initiating proactive unload.")
            # Proactively clear PromptExecutor caches ahead of major DisTorch2 load (Phase 1)
            trigger_executor_cache_reset(reason="proactive_distorch_load", force=False)
            if not hasattr(mm, 'current_loaded_models'):
                raise AttributeError("comfy.model_management is missing 'current_loaded_models'. Proactive unload check failed.")

            needed_patchers = incoming_patchers
            # Need-based free on compute device only (scale-aware; core-aligned)
            dev_str = incoming_compute_device or (next(iter(incoming_allowed_devices)) if incoming_allowed_devices else None)
            freed_bytes = 0
            to_unload_indices = []
            unload_summaries = []
            if dev_str is not None:
                dev_obj = torch.device(dev_str)
                free_now = mm.get_free_memory(dev_obj)
                try:
                    free_now_val = free_now[0] if isinstance(free_now, tuple) else free_now
                except Exception:
                    free_now_val = free_now
                # Use core-aligned immediate needs: planned vs. memory_required vs. minimum_memory_required
                effective_needed = max(incoming_compute_planned_bytes or 0, memory_required or 0, minimum_memory_required or 0)
                need_bytes = max(0, effective_needed - (free_now_val or 0))
                logger.mgpu_mm_log(f"Need calc on {dev_str}: effective_needed={effective_needed/(1024**3):.2f}GB, free_now={((free_now_val or 0)/(1024**3)):.2f}GB, need_bytes={need_bytes/(1024**3):.2f}GB")
                if need_bytes > 0:
                    logger.mgpu_mm_log(f"Need-based unload on {dev_str}: need ~{need_bytes/(1024**3):.2f}GB")
                    # Build candidates on this device only, excluding needed patchers
                    candidates = []
                    for idx, lm_cur in enumerate(mm.current_loaded_models):
                        mp_cur = getattr(lm_cur, 'model', None)
                        if mp_cur is None or mp_cur in needed_patchers:
                            continue
                        if str(getattr(lm_cur, "device", "")) != dev_str:
                            continue
                        size_cur = 0
                        if hasattr(lm_cur, 'model_memory'):
                            try:
                                size_cur = lm_cur.model_memory()
                            except Exception:
                                size_cur = 0
                        if size_cur <= 0 and hasattr(mp_cur, 'model_size'):
                            size_cur = mp_cur.model_size()
                        candidates.append((size_cur, idx, lm_cur, mp_cur))
                    # Sort by size descending
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    for size_cur, idx, lm_cur, mp_cur in candidates:
                        model_name = type(getattr(mp_cur, 'model', mp_cur)).__name__
                        logger.mgpu_mm_log(f"Unloading model on {dev_str}: {model_name} (~{size_cur/(1024**3):.2f}GB)")
                        success = False
                        if hasattr(lm_cur, 'model_unload'):
                            success = lm_cur.model_unload(memory_to_free=None, unpatch_weights=True)
                        if success:
                            to_unload_indices.append(idx)
                            unload_summaries.append(f"{model_name}:{size_cur/(1024**3):.2f}GB")
                            freed_bytes += size_cur
                            if freed_bytes >= need_bytes:
                                break

            # Remove from management list and clear caches
            unloaded_count = 0
            for idx in sorted(to_unload_indices, reverse=True):
                mm.current_loaded_models.pop(idx)
                unloaded_count += 1

            if unloaded_count > 0:
                logger.mgpu_mm_log(f"Proactively unloaded {unloaded_count} large model(s): {', '.join(unload_summaries)}")
                logger.mgpu_mm_log("Performing multi-device cache clear after proactive unload")
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
                        logger.mgpu_mm_log(f"No unloads; 25% torch-cache rule triggered on: {', '.join(triggered)}. Calling soft_empty_cache()")
                        mm.soft_empty_cache(force=True)
                    else:
                        logger.mgpu_mm_log("No unloads; 25% torch-cache rule not met on DisTorch devices; skipping cache clear")
                else:
                    logger.mgpu_mm_log("No unload candidates matched criteria and either HIGH_VRAM or no DisTorch devices; skipping cache clear")
        elif incoming_is_distorch:
            logger.mgpu_mm_log("Incoming DisTorch2 model requires 0.00GB; skipping proactive unload")

        # Memory Logging
        multigpu_memory_log("patched_load_models_gpu", "pre-original-call")
        result = original_load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)
        multigpu_memory_log("patched_load_models_gpu", "post-original-call")

        return result

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
