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
# GC Anchor System for Model Retention Testing
# ==========================================================================================

# Global anchor set to prevent GC of models with keep_loaded=True
_MGPU_RETENTION_ANCHORS = set()

def add_retention_anchor(model_patcher, reason="keep_loaded"):
    """Add a model patcher to the GC anchor set to prevent premature garbage collection"""
    if model_patcher is not None:
        _MGPU_RETENTION_ANCHORS.add(model_patcher)
        model_name = type(getattr(model_patcher, 'model', model_patcher)).__name__
        logger.mgpu_mm_log(f"[GC_ANCHOR] Added retention anchor for {model_name}, reason: {reason}, total anchors: {len(_MGPU_RETENTION_ANCHORS)}")

def remove_retention_anchor(model_patcher, reason="cleanup"):
    """Remove a model patcher from the GC anchor set"""
    if model_patcher is not None and model_patcher in _MGPU_RETENTION_ANCHORS:
        _MGPU_RETENTION_ANCHORS.discard(model_patcher)
        model_name = type(getattr(model_patcher, 'model', model_patcher)).__name__
        logger.mgpu_mm_log(f"[GC_ANCHOR] Removed retention anchor for {model_name}, reason: {reason}, total anchors: {len(_MGPU_RETENTION_ANCHORS)}")

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

def prune_distorch_stores():
    """Prune stale allocation/settings entries not tied to active models."""
    multigpu_memory_log("distorch_prune", "start")
    active_hashes_v2 = set()
    active_hashes_v1 = set()
    
    logger.mgpu_mm_log(f"[PRUNE_DEBUG] Starting prune - current_loaded_models count: {len(mm.current_loaded_models)}")
    
    for i, lm in enumerate(mm.current_loaded_models):
        mp = lm.model
        if mp is not None:
            try:
                hash_v2 = create_safetensor_model_hash(mp, "prune_check_v2")
                hash_v1 = create_model_hash(mp, "prune_check_v1")
                active_hashes_v2.add(hash_v2)
                active_hashes_v1.add(hash_v1)
                
                model_name = type(getattr(mp, 'model', mp)).__name__
                keep_loaded = getattr(getattr(mp, 'model', None), '_mgpu_keep_loaded', False)
                has_v2_alloc = hash_v2 in safetensor_allocation_store
                logger.mgpu_mm_log(f"[PRUNE_DEBUG] Model {i}: {model_name}, keep_loaded={keep_loaded}, hash={hash_v2[:8]}, has_v2_allocation={has_v2_alloc}")
            except Exception as e:
                logger.mgpu_mm_log(f"[PRUNE_DEBUG] Model {i}: Error getting hash - {e}")

    logger.mgpu_mm_log(f"[PRUNE_DEBUG] Active hashes V2: {len(active_hashes_v2)}, Store has: {len(safetensor_allocation_store)}")

    # V1 pruning
    stale_v1 = set(model_allocation_store.keys()) - active_hashes_v1
    if stale_v1:
        logger.mgpu_mm_log(f"[MultiGPU_Memory_Management] Pruning {len(stale_v1)} DisTorch V1 entries")
        for k in stale_v1:
            del model_allocation_store[k]

    # V2 pruning with diagnostics
    for store, name in ((safetensor_allocation_store, "allocation"), (safetensor_settings_store, "settings")):
        stale_v2 = set(store.keys()) - active_hashes_v2
        if stale_v2:
            logger.mgpu_mm_log(f"[MultiGPU_Memory_Management] Would prune {len(stale_v2)} V2 {name} entries: {[h[:8] for h in list(stale_v2)[:5]]}")
            for k in stale_v2:
                del store[k]
        else:
            logger.mgpu_mm_log(f"[PRUNE_DEBUG] No stale {name} entries to prune")
    
    logger.mgpu_mm_log(f"[PRUNE_DEBUG] After pruning - V2 allocation store has: {len(safetensor_allocation_store)} entries")
    multigpu_memory_log("distorch_prune", "end")

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

def clear_memory_snapshot_history():
    """Clear stored memory snapshot history"""
    multigpu_memory_log("mem_mgmt", "pre-history-clear")
    _MEM_SNAPSHOT_LAST.clear()
    _MEM_SNAPSHOT_SERIES.clear()
    logger.debug("[MultiGPU_Memory_Management] Memory snapshot history cleared")
    multigpu_memory_log("mem_mgmt", "post-history-clear")

# ==========================================================================================
# ModelPatcher Lifecycle Tracking
# ==========================================================================================

_MGPU_TRACKED_MODELPATCHERS = weakref.WeakSet()

def track_modelpatcher(model_patcher):
    """Register ModelPatcher for lifecycle tracking"""
    if isinstance(model_patcher, comfy.model_patcher.ModelPatcher):
        if model_patcher not in _MGPU_TRACKED_MODELPATCHERS:
            _MGPU_TRACKED_MODELPATCHERS.add(model_patcher)
            logger.debug(f"[MultiGPU_Lifecycle] Tracking ModelPatcher {id(model_patcher)} (tracked={len(_MGPU_TRACKED_MODELPATCHERS)})")

def log_tracked_modelpatchers_status(tag="checkpoint"):
    """Log count and estimated CPU RAM for tracked ModelPatchers"""
    alive_count = len(_MGPU_TRACKED_MODELPATCHERS)
    total_cpu_memory_mb = 0.0
    
    for patcher in list(_MGPU_TRACKED_MODELPATCHERS):
        if hasattr(patcher, "model") and patcher.model is not None:
            for param in patcher.model.parameters():
                if getattr(param, "device", torch.device("cpu")).type == "cpu":
                    total_cpu_memory_mb += (param.nelement() * param.element_size()) / (1024.0 * 1024.0)
    
    logger.warning(f"[MultiGPU_Lifecycle] [{tag}] Tracked ModelPatchers={alive_count}, approx CPU RAM={total_cpu_memory_mb:.2f} MB")

def analyze_cpu_memory_leaks():
    """Diagnostic: scan referrers of tracked ModelPatchers when memory is high"""
    vm = psutil.virtual_memory()
    patchers = list(_MGPU_TRACKED_MODELPATCHERS)
    
    if len(patchers) <= 5 and vm.percent <= 80.0:
        logger.debug(f"[MultiGPU_Leak_Analyzer] Skipping analysis. Normal conditions: patchers={len(patchers)}, memory={vm.percent:.1f}%")
        return
        
    logger.warning(f"[MultiGPU_Leak_Analyzer] High pressure detected: patchers={len(patchers)}, cpu_mem={vm.percent:.1f}%. Analyzing referrers.")
    
    for i, patcher in enumerate(patchers[:5]):
        referrers = gc.get_referrers(patcher)
        logger.warning(f"[MultiGPU_Leak_Analyzer] Patcher #{i} id={id(patcher)} referrers={len(referrers)}")
        
        for j, ref in enumerate(referrers[:10]):
            rtype = type(ref).__name__
            rmod = getattr(type(ref), "__module__", "unknown")
            if isinstance(ref, dict):
                logger.warning(f"  Ref {j}: dict(len={len(ref)}) mod={rmod}")
            elif isinstance(ref, list):
                logger.warning(f"  Ref {j}: list(len={len(ref)}) mod={rmod}")
            else:
                logger.warning(f"  Ref {j}: {rtype} mod={rmod}")

# ==========================================================================================
# Memory Management and Cleanup
# ==========================================================================================

CPU_MEMORY_THRESHOLD_PERCENT = 85.0
CPU_RESET_HYSTERESIS_PERCENT = 5.0
_last_cpu_usage_at_reset = 0.0

def try_malloc_trim():
    """Return freed heap memory to OS (Linux/glibc)"""
    if platform.system() != "Linux":
        return
        
    libc = ctypes.CDLL("libc.so.6")
    if not hasattr(libc, "malloc_trim"):
        return
        
    logger.info("[MultiGPU_Memory_Management] malloc_trim(0) begin")
    multigpu_memory_log("mem_mgmt", "pre-malloc-trim")
    
    result = libc.malloc_trim(0)
    
    multigpu_memory_log("mem_mgmt", "post-malloc-trim")
    if result == 1:
        logger.info("[MultiGPU_Memory_Management] malloc_trim(0) released memory")
    else:
        logger.debug("[MultiGPU_Memory_Management] malloc_trim(0) no release")

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

    analyze_cpu_memory_leaks()
    prune_distorch_stores()
    clear_memory_snapshot_history()

    prompt_server.prompt_queue.set_flag("free_memory", True)
    logger.debug("[MultiGPU_Memory_Management] 'free_memory' flag set")

    vm = psutil.virtual_memory()
    _last_cpu_usage_at_reset = vm.percent

    try_malloc_trim()
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
    """
    Mirror ComfyUI-Manager 'Free model and node cache' by setting both flags:
    unload_models=True and free_memory=True
    """
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
# Core Patching: unload_all_models with keep_loaded retention
# ==========================================================================================

if hasattr(mm, 'unload_all_models') and not hasattr(mm.unload_all_models, '_mgpu_keep_loaded_patched'):
    logger.info("[MultiGPU Core Patching] Patching mm.unload_all_models to respect keep_loaded flag for DisTorch models")
    
    _mgpu_original_unload_all_models = mm.unload_all_models
    
    def _mgpu_patched_unload_all_models():
        """
        Patched mm.unload_all_models that preserves DisTorch models with _mgpu_keep_loaded=True.
        All other models (including DisTorch models without the flag) unload normally.
        """
        logger.mgpu_mm_log(f"[UNLOAD_DEBUG] Patched unload_all_models called - initial model count: {len(mm.current_loaded_models)}")
        
        # Direct approach: iterate through loaded models and selectively unload
        models_to_unload = []
        kept_models = []
        
        for i, lm in enumerate(mm.current_loaded_models):
            mp = lm.model  # weakref call to ModelPatcher
            if mp is not None and hasattr(mp, 'model'):
                # Check if this is a DisTorch model with keep_loaded flag
                should_retain = getattr(mp.model, '_mgpu_keep_loaded', True)
                model_name = type(getattr(mp, 'model', mp)).__name__
                logger.mgpu_mm_log(f"[UNLOAD_DEBUG] Model {i}: {model_name}, keep_loaded={should_retain}")
                
                # Retain models that either:
                # 1. Are non-DisTorch models (missing _mgpu_keep_loaded attribute)
                # 2. Are DisTorch models with keep_loaded=True

                if should_retain:
                    kept_models.append(lm)
                    logger.mgpu_mm_log(f"[UNLOAD_DEBUG] Adding to kept_models: {model_name}")
                    # GC ANCHOR TEST: Prevent premature GC of clone patchers
                    add_retention_anchor(mp, "keep_loaded_test")
                else:
                    models_to_unload.append(lm)
            else:
                logger.mgpu_mm_log(f"[UNLOAD_DEBUG] Model {i}: ModelPatcher is None or missing model attribute")
                models_to_unload.append(lm)
        
        logger.mgpu_mm_log(f"[UNLOAD_DEBUG] Final counts - kept_models: {len(kept_models)}, models_to_unload: {len(models_to_unload)}")
        
        if kept_models:
            logger.mgpu_mm_log(f"Found {len(kept_models)} model(s) to retain, unloading {len(models_to_unload)} model(s)")
            
            # Unload models that don't have keep_loaded flag
            for lm in models_to_unload:
                try:
                    lm.model_unload(unpatch_weights=True)
                    logger.debug(f"Unloaded model: {type(lm.model.model).__name__ if lm.model else 'Unknown'}")
                except Exception as e:
                    logger.warning(f"Error unloading model: {e}")
            
            # Remove unloaded models from current_loaded_models
            mm.current_loaded_models = kept_models
            logger.mgpu_mm_log(f"[UNLOAD_DEBUG] Updated mm.current_loaded_models, new count: {len(mm.current_loaded_models)}")
            logger.mgpu_mm_log(f"Successfully retained {len(kept_models)} model(s) during unload")
        else:
            logger.mgpu_mm_log("No models with keep_loaded=True found - delegating to original unload_all_models")
            _mgpu_original_unload_all_models()
    
    mm.unload_all_models = _mgpu_patched_unload_all_models
    mm.unload_all_models._mgpu_keep_loaded_patched = True
    logger.info("[MultiGPU Core Patching] mm.unload_all_models patched successfully")
else:
    if not hasattr(mm, 'unload_all_models'):
        logger.warning("[MultiGPU Core Patching] mm.unload_all_models not found - cannot patch keep_loaded retention")
    else:
        logger.debug("[MultiGPU Core Patching] mm.unload_all_models already patched for keep_loaded - skipping")
