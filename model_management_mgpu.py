"""
Model Management Extensions for MultiGPU
Extends ComfyUI's model management with multi-device capabilities and lifecycle tracking.
"""

import torch
import logging
import hashlib
import psutil
import comfy.model_management as mm
import gc
from datetime import datetime, timezone
import server
import weakref
import platform
import ctypes
import comfy.model_patcher
from collections import defaultdict



logger = logging.getLogger("MultiGPU")

# ==========================================================================================
# GC Anchor System for Model Retention
# ==========================================================================================

# Global anchor set to prevent GC of models during selective unload
_MGPU_RETENTION_ANCHORS = set()

def add_retention_anchor(model_patcher, reason="keep_loaded"):
    """Add a model patcher to the GC anchor set to prevent premature garbage collection"""
    if model_patcher is not None:
        _MGPU_RETENTION_ANCHORS.add(model_patcher)
        model_name = type(getattr(model_patcher, 'model', model_patcher)).__name__
        logger.mgpu_mm_log(f"[GC_ANCHOR] Added retention anchor for {model_name}, reason: {reason}, total anchors: {len(_MGPU_RETENTION_ANCHORS)}")

def clear_all_retention_anchors(reason="manual_clear"):
    """Clear all retention anchors"""
    count = len(_MGPU_RETENTION_ANCHORS)
    _MGPU_RETENTION_ANCHORS.clear()
    logger.mgpu_mm_log(f"[GC_ANCHOR] Cleared all {count} retention anchors, reason: {reason}")

# ==========================================================================================
# Model Analysis and Store Management (DisTorch V1 & V2)
# ==========================================================================================

# DisTorch V2 SafeTensor stores
safetensor_allocation_store = {}
safetensor_settings_store = {}

# DisTorch V1 GGUF stores (backwards compatibility)
model_allocation_store = {}

def create_safetensor_model_hash(model, caller):
    """Create a unique hash for a safetensor model to track allocations"""
    if hasattr(model, 'model'):
        actual_model = model.model
        model_type = type(actual_model).__name__
        model_size = model.model_size() if hasattr(model, 'model_size') else sum(p.numel() * p.element_size() for p in actual_model.parameters())
        first_layers = str(list(model.model_state_dict().keys() if hasattr(model, 'model_state_dict') else actual_model.state_dict().keys())[:3])
    else:
        model_type = type(model).__name__
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        first_layers = str(list(model.state_dict().keys())[:3])
    
    identifier = f"{model_type}_{model_size}_{first_layers}"
    final_hash = hashlib.sha256(identifier.encode()).hexdigest()
    logger.debug(f"[MultiGPU DisTorch V2] Created hash for {caller}: {final_hash[:8]}...")
    return final_hash

def create_model_hash(model, caller):
    """Create a unique hash for a GGUF model to track allocations (DisTorch V1)"""
    model_type = type(model.model).__name__
    model_size = model.model_size()
    first_layers = str(list(model.model_state_dict().keys())[:3])
    identifier = f"{model_type}_{model_size}_{first_layers}"
    final_hash = hashlib.sha256(identifier.encode()).hexdigest()
    logger.debug(f"[MultiGPU_DisTorch_HASH] Created hash for {caller}: {final_hash[:8]}...")
    return final_hash

# ==========================================================================================
# Memory Logging Infrastructure
# ==========================================================================================

_MEM_SNAPSHOT_LAST = {}
_MEM_SNAPSHOT_SERIES = {}

def _capture_memory_snapshot():
    """Capture memory snapshot for CPU and all devices"""
    # Import here to avoid circular dependency
    from .device_utils import get_device_list
    
    snapshot = {}
    
    # CPU
    vm = psutil.virtual_memory()
    snapshot["cpu"] = (vm.used, vm.total)
    
    # GPU devices
    devices = [d for d in get_device_list() if d != "cpu"]
    for dev_str in devices:
        device = torch.device(dev_str)
        total = mm.get_total_memory(device)
        free_info = mm.get_free_memory(device, torch_free_too=True)
        system_free = free_info[0] if isinstance(free_info, tuple) else free_info
        used = max(0, total - system_free)
        snapshot[dev_str] = (used, total)

    return snapshot

def multigpu_memory_log(identifier, tag):
    """Record timestamped memory snapshot with clean aligned logging"""
    if identifier == "print_summary":
        for id_key in sorted(_MEM_SNAPSHOT_SERIES.keys()):
            series = _MEM_SNAPSHOT_SERIES[id_key]
            logger.mgpu_mm_log(f"=== memory summary: {id_key} ===")
            for ts, tag_name, snap in series:
                parts = []
                cpu_used, cpu_total = snap.get("cpu", (0, 0))
                parts.append(f"cpu|{cpu_used/(1024**3):.2f}")
                for dev in sorted([k for k in snap.keys() if k != "cpu"]):
                    used, total = snap[dev]
                    parts.append(f"{dev}|{used/(1024**3):.2f}")
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                tag_padded = f"{id_key}_{tag_name}".ljust(35)
                logger.mgpu_mm_log(f"{ts_str} {tag_padded} {' '.join(parts)}")
        return

    ts = datetime.now(timezone.utc)
    curr = _capture_memory_snapshot()
    
    # Store in series
    if identifier not in _MEM_SNAPSHOT_SERIES:
        _MEM_SNAPSHOT_SERIES[identifier] = []
    _MEM_SNAPSHOT_SERIES[identifier].append((ts, tag, curr))
    
    # Clean aligned format: timestamp + padded tag + memory values
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    tag_padded = f"{identifier}_{tag}".ljust(35)
    
    parts = []
    cpu_used, _ = curr.get("cpu", (0, 0))
    parts.append(f"cpu|{cpu_used/(1024**3):.2f}")
    
    for dev in sorted([k for k in curr.keys() if k != "cpu"]):
        used, _ = curr[dev]
        parts.append(f"{dev}|{used/(1024**3):.2f}")
    
    logger.mgpu_mm_log(f"{ts_str} {tag_padded} {' '.join(parts)}")
    
    _MEM_SNAPSHOT_LAST[identifier] = (tag, curr)


# ==========================================================================================
# Memory Management and Cleanup
# ==========================================================================================

CPU_MEMORY_THRESHOLD_PERCENT = 85.0
CPU_RESET_HYSTERESIS_PERCENT = 5.0
_last_cpu_usage_at_reset = 0.0


def trigger_executor_cache_reset(reason="policy", force=False):
    """Trigger PromptExecutor.reset() by setting 'free_memory' flag"""
    global _last_cpu_usage_at_reset

    prompt_server = server.PromptServer.instance
    if prompt_server is None:
        logger.debug("[MultiGPU_Memory_Management] PromptServer not initialized")
        return

    if prompt_server.prompt_queue.currently_running and not force:
        logger.debug(f"[MultiGPU_Memory_Management] Skipping reset during execution (reason: {reason})")
        return

    multigpu_memory_log("executor_reset", f"pre-trigger ({reason})")
    logger.info(f"[MultiGPU_Memory_Management] Triggering PromptExecutor cache reset. Reason: {reason}")

    prompt_server.prompt_queue.set_flag("free_memory", True)
    logger.debug("[MultiGPU_Memory_Management] 'free_memory' flag set")

    vm = psutil.virtual_memory()
    _last_cpu_usage_at_reset = vm.percent

    multigpu_memory_log("executor_reset", f"post-trigger ({reason})")

def check_cpu_memory_threshold(threshold_percent=CPU_MEMORY_THRESHOLD_PERCENT):
    """Check CPU memory and trigger reset if threshold exceeded"""
    if server.PromptServer.instance is None:
        return

    if server.PromptServer.instance.prompt_queue.currently_running:
        return

    vm = psutil.virtual_memory()
    current_usage = vm.percent

    if current_usage > threshold_percent:
        if current_usage > (_last_cpu_usage_at_reset + CPU_RESET_HYSTERESIS_PERCENT):
            logger.warning(f"[MultiGPU_Memory_Monitor] CPU usage ({current_usage:.1f}%) exceeds threshold ({threshold_percent:.1f}%)")
            multigpu_memory_log("cpu_monitor", f"trigger:{current_usage:.1f}pct")
            trigger_executor_cache_reset(reason="cpu_threshold_exceeded", force=False)
        else:
            logger.debug(f"[MultiGPU_Memory_Monitor] CPU usage high ({current_usage:.1f}%) but within hysteresis")
            multigpu_memory_log("cpu_monitor", f"skip_hysteresis:{current_usage:.1f}pct")

def force_full_system_cleanup(reason="manual", force=True):
    """Mirror ComfyUI-Manager 'Free model and node cache' by setting unload_models=True and free_memory=True flags."""
    vm = psutil.virtual_memory()
    pre_cpu = vm.used
    pre_models = len(mm.current_loaded_models)

    multigpu_memory_log("full_cleanup", f"start:{reason}")
    logger.mgpu_mm_log(f"[ManagerMatch] Requesting cleanup (reason={reason}) | pre_models={pre_models}, cpu_used_gib={pre_cpu/(1024**3):.2f}")

    if server.PromptServer.instance is not None:
        pq = server.PromptServer.instance.prompt_queue
        if (not pq.currently_running) or force:
            pq.set_flag("unload_models", True)
            pq.set_flag("free_memory", True)
            logger.mgpu_mm_log("[ManagerMatch] Flags set: unload_models=True, free_memory=True")
        else:
            logger.mgpu_mm_log("[ManagerMatch] Skipped - execution active and force=False")

    vm = psutil.virtual_memory()
    post_cpu = vm.used
    post_models = len(mm.current_loaded_models)
    delta_cpu_mb = (post_cpu - pre_cpu) / (1024**2)

    multigpu_memory_log("full_cleanup", f"requested:{reason}")
    summary = f"[ManagerMatch] Cleanup requested (reason={reason}) | models {pre_models}->{post_models}, cpu_delta_mb={delta_cpu_mb:.2f}"
    logger.mgpu_mm_log(summary)
    return summary

# ==========================================================================================
# Core Patching: unload_all_models
# ==========================================================================================

if not hasattr(mm.unload_all_models, '_mgpu_eject_distorch_patched'):
    logger.info("[MultiGPU Core Patching] Patching mm.unload_all_models for DisTorch2 ejection support")
    
    _mgpu_original_unload_all_models = mm.unload_all_models
    
    def _mgpu_patched_unload_all_models():
        """Patched mm.unload_all_models with selective ejection support and comprehensive diagnostics."""

        logger.mgpu_mm_log(f"[UNLOAD_START] Patched unload_all_models called - initial model count: {len(mm.current_loaded_models)}")

        # Check if there are any DisTorch models that want to be unloaded
        has_distorch_to_unload = any(
            (hasattr(lm.model, '_mgpu_unload_distorch_model') and lm.model._mgpu_unload_distorch_model) or
            (hasattr(getattr(lm.model, 'model', None), '_mgpu_unload_distorch_model') and lm.model.model._mgpu_unload_distorch_model)
            for lm in mm.current_loaded_models
            if lm.model is not None
        )
        
        if not has_distorch_to_unload:
            logger.mgpu_mm_log("No DisTorch models requesting unload - clearing anchors and delegating to original unload_all_models")
            clear_all_retention_anchors(reason="no_selective_unload_needed")
            _mgpu_original_unload_all_models()
            return

        # Direct approach: iterate through loaded models and selectively unload
        models_to_unload = []
        kept_models = []
        
        for i, lm in enumerate(mm.current_loaded_models):
            mp = lm.model  # weakref call to ModelPatcher
            
            # DIAGNOSTIC: Log full object chain
            lm_id = id(lm)
            mp_id = id(mp)
            inner_model = getattr(mp, 'model', None)
            inner_model_id = id(inner_model) if inner_model else None
            inner_model_name = type(inner_model).__name__ if inner_model else "None"
            
            # Format inner_model_id properly for f-string
            inner_id_str = f"0x{inner_model_id:x}" if inner_model_id is not None else "None"
            
            logger.mgpu_mm_log(f"[OBJECT_CHAIN_READ] Model {i}: lm_id=0x{lm_id:x}, mp_id=0x{mp_id:x}, inner_model_id={inner_id_str}, inner_model_type={inner_model_name}")
            
            # FIX: Check flag on ModelPatcher (where it was set), not on inner model
            # OLD BUG: unload_distorch_model = getattr(mp.model, '_mgpu_unload_distorch_model', False)
            # NEW FIX: Check both locations to see which one has the flag
            flag_on_mp = getattr(mp, '_mgpu_unload_distorch_model', None)
            flag_on_inner = getattr(mp.model, '_mgpu_unload_distorch_model', None) if inner_model else None
            
            logger.mgpu_mm_log(f"[FLAG_CHECK] Model {i} ({inner_model_name}): flag_on_mp={flag_on_mp}, flag_on_inner={flag_on_inner}")
            
            # Use whichever location has the flag (for backwards compatibility during transition)
            if flag_on_mp is not None:
                unload_distorch_model = flag_on_mp
                logger.mgpu_mm_log(f"[FLAG_SOURCE] Using flag from ModelPatcher (mp_id=0x{mp_id:x})")
            elif flag_on_inner is not None:
                unload_distorch_model = flag_on_inner
                logger.mgpu_mm_log(f"[FLAG_SOURCE] Using flag from inner model (inner_model_id={inner_id_str})")
            else:
                unload_distorch_model = False
                logger.mgpu_mm_log(f"[FLAG_SOURCE] No flag found - defaulting to False (keep loaded)")
            
            logger.mgpu_mm_log(f"[DECISION] Model {i} ({inner_model_name}): unload_distorch_model={unload_distorch_model}")
            
            if unload_distorch_model:
                models_to_unload.append(lm)
                logger.mgpu_mm_log(f"[CATEGORIZE] Model {i} ({inner_model_name}) → models_to_unload")
            else:
                kept_models.append(lm)
                add_retention_anchor(mp, "keep_loaded_protection")
                logger.mgpu_mm_log(f"[CATEGORIZE] Model {i} ({inner_model_name}) → kept_models")

        # After the kept_models/models_to_unload evaluation
        logger.mgpu_mm_log(f"[CATEGORIZE_SUMMARY] kept_models: {len(kept_models)}, models_to_unload: {len(models_to_unload)}, total: {len(mm.current_loaded_models)}")
        
        if len(kept_models) == len(mm.current_loaded_models):
            # All models are meant to be kept - no DisTorch selective unloading needed
            logger.mgpu_mm_log("[DELEGATION] All models flagged to be kept - delegating to standard unload_all_models")
            _mgpu_original_unload_all_models()
            return

        if kept_models:
            logger.mgpu_mm_log(f"[SELECTIVE_UNLOAD] Proceeding with selective unload: retaining {len(kept_models)}, unloading {len(models_to_unload)}")

            # Unload models flagged for unload
            for lm in models_to_unload:
                try:
                    model_name = type(lm.model.model).__name__ if lm.model and hasattr(lm.model, 'model') else 'Unknown'
                    logger.mgpu_mm_log(f"[UNLOAD_EXECUTE] Unloading model: {model_name} (lm_id=0x{id(lm):x})")
                    lm.model_unload(unpatch_weights=True)
                except Exception as e:
                    logger.warning(f"[UNLOAD_ERROR] Error unloading model: {e}")

            # WEAKREF TRACKING: Attach weakref callbacks to prove if kept models are GC'd
            def model_deleted_callback(ref, model_name, model_id):
                logger.mgpu_mm_log(f"[WEAKREF_DELETED] Kept model GARBAGE COLLECTED: {model_name} (id=0x{model_id:x})")
            
            for i, lm in enumerate(kept_models):
                mp = lm.model
                inner_model = getattr(mp, 'model', None)
                model_name = type(inner_model).__name__ if inner_model else 'Unknown'
                model_id = id(lm)
                weakref.ref(lm, lambda ref, name=model_name, mid=model_id: model_deleted_callback(ref, name, mid))
                logger.mgpu_mm_log(f"[WEAKREF_ATTACHED] Tracking kept model {i}: {model_name} (lm_id=0x{model_id:x}, mp_id=0x{id(mp):x})")

            # Remove unloaded models from current_loaded_models
            mm.current_loaded_models = kept_models
            logger.mgpu_mm_log(f"[SELECTIVE_COMPLETE] Updated mm.current_loaded_models, new count: {len(mm.current_loaded_models)}")
            logger.mgpu_mm_log(f"[SELECTIVE_COMPLETE] mm.current_loaded_models id: 0x{id(mm.current_loaded_models):x}")
            
            # DIAGNOSTIC: Log what's remaining
            for i, lm in enumerate(mm.current_loaded_models):
                mp = lm.model
                inner_model = getattr(mp, 'model', None)
                model_name = type(inner_model).__name__ if inner_model else "None"
                logger.mgpu_mm_log(f"[REMAINING_MODEL] {i}: {model_name} (lm_id=0x{id(lm):x}, mp_id=0x{id(mp):x})")
        else:
            logger.mgpu_mm_log("[DELEGATION] No models with keep_loaded=True found - delegating to original unload_all_models")
            _mgpu_original_unload_all_models()
    
    mm.unload_all_models = _mgpu_patched_unload_all_models
    mm.unload_all_models._mgpu_eject_distorch_patched = True
    logger.info("[MultiGPU Core Patching] mm.unload_all_models patched successfully")
else:
    logger.debug("[MultiGPU Core Patching] mm.unload_all_models already patched - skipping")
