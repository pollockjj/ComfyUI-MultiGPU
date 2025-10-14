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
from collections import defaultdict



logger = logging.getLogger("MultiGPU")


# ==========================================================================================
# Model Analysis and Store Management (DisTorch V1 & V2)
# ==========================================================================================

# DisTorch V1 GGUF stores (backwards compatibility)
model_allocation_store = {}

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
