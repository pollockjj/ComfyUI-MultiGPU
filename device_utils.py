"""
Device detection, management, and inspection utilities for ComfyUI-MultiGPU.
Single source of truth for all device enumeration, compatibility checks, and state inspection.
Handles all device types supported by ComfyUI core.
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
import sys
import comfy.model_patcher

# DisTorch stores for pruning/diagnostics
from .distorch_2 import (
    safetensor_allocation_store,
    safetensor_settings_store,
    create_safetensor_model_hash,
)

# Optional DisTorch v1 store support
try:
    from .distorch import (
        model_allocation_store,
        create_model_hash,
    )
except Exception:
    model_allocation_store = {}
    create_model_hash = None

logger = logging.getLogger("MultiGPU")

# Module-level cache for device list (populated once on first call)
_DEVICE_LIST_CACHE = None

# ==========================================================================================
# Executor Cache Management and CPU Monitoring (Phases 1, 2, 3)
# ==========================================================================================

# Configuration for CPU Monitoring (Phase 2)
CPU_MEMORY_THRESHOLD_PERCENT = 85.0
# Hysteresis: Only trigger again if usage increased by this amount since the last reset.
CPU_RESET_HYSTERESIS_PERCENT = 5.0
_last_cpu_usage_at_reset = 0.0

def clear_memory_snapshot_history():
    """Clears the stored memory snapshot history. (Phase 3)"""
    # Logging integration
    multigpu_memory_log("mem_mgmt", "pre-history-clear")

    # Snapshot globals exist in this module; operate safely in case of reload
    if '_MEM_SNAPSHOT_LAST' in globals():
        globals()['_MEM_SNAPSHOT_LAST'].clear()
    if '_MEM_SNAPSHOT_SERIES' in globals():
        globals()['_MEM_SNAPSHOT_SERIES'].clear()
    logger.debug("[MultiGPU_Memory_Management] Memory snapshot history cleared.")

    # Logging integration
    multigpu_memory_log("mem_mgmt", "post-history-clear")

def trigger_executor_cache_reset(reason="policy", force=False):
    """
    (Phase 1/2 Core) Triggers PromptExecutor.reset() by setting the 'free_memory' flag.
    Releases CPU-side references held by execution caches.
    """
    global _last_cpu_usage_at_reset

    # Ensure PromptServer singleton is available
    if server.PromptServer.instance is None:
        logger.debug("[MultiGPU_Memory_Management] PromptServer instance not yet initialized.")
        return

    prompt_server = server.PromptServer.instance

    # Stability guard: Avoid during active execution unless forced
    if prompt_server.prompt_queue.currently_running and not force:
        logger.debug(f"[MultiGPU_Memory_Management] Skipping Executor Cache Reset during active prompt execution (Reason: {reason}).")
        return

    multigpu_memory_log("executor_reset", f"pre-trigger ({reason})")
    logger.info(f"[MultiGPU_Memory_Management] Triggering PromptExecutor cache reset (e.reset()). Reason: {reason}")

    # Diagnostics and store pruning prior to reset
    analyze_cpu_memory_leaks(force=force)
    prune_distorch_stores()

    # Phase 3: Clear internal snapshot history as the context is resetting
    clear_memory_snapshot_history()

    # Set the flag on the prompt queue (ComfyUI core mechanism)
    prompt_server.prompt_queue.set_flag("free_memory", True)
    logger.debug("[MultiGPU_Memory_Management] 'free_memory' flag set.")

    # Update usage baseline for hysteresis
    vm = psutil.virtual_memory()
    _last_cpu_usage_at_reset = vm.percent

    # Attempt to return freed memory to OS
    try_malloc_trim()

    multigpu_memory_log("executor_reset", f"post-trigger ({reason})")


def _cpu_used_bytes():
    try:
        vm = psutil.virtual_memory()
        return vm.used
    except Exception:
        return 0


def force_full_system_cleanup(reason="manual", force=True):
    """
    Mirror ComfyUI-Manager 'Free model and node cache' semantics:
    - Only set unload_models=True and free_memory=True flags on the PromptQueue
    - The prompt worker (main.py) performs unload/reset/GC
    """
    pre_cpu = _cpu_used_bytes()
    pre_models = len(getattr(mm, "current_loaded_models", []))

    multigpu_memory_log("full_cleanup", f"start:{reason}")
    logger.mgpu_mm_log(f"[ManagerMatch] Requesting flags-only cleanup (reason={reason}) | pre_models={pre_models}, cpu_used_gib={pre_cpu/(1024**3):.2f}")

    try:
        if server.PromptServer.instance is not None:
            pq = server.PromptServer.instance.prompt_queue
            # Respect currently_running unless forced
            if (not pq.currently_running) or force:
                pq.set_flag("unload_models", True)
                pq.set_flag("free_memory", True)
                logger.mgpu_mm_log("[ManagerMatch] Flags set: unload_models=True, free_memory=True")
            else:
                logger.mgpu_mm_log("[ManagerMatch] Skipped setting flags due to active execution and force=False")
    except Exception as e:
        logger.mgpu_mm_log(f"[ManagerMatch] Failed to set flags: {e}")

    post_cpu = _cpu_used_bytes()
    post_models = len(getattr(mm, "current_loaded_models", []))
    delta_cpu_mb = (post_cpu - pre_cpu) / (1024**2)

    multigpu_memory_log("full_cleanup", f"requested:{reason}")
    summary = (
        f"[ManagerMatch] Flags-only cleanup requested (reason={reason}) | "
        f"models {pre_models}->{post_models} (no immediate unload), cpu_delta_mb={delta_cpu_mb:.2f}"
    )
    logger.mgpu_mm_log(summary)
    return summary

def check_cpu_memory_threshold(threshold_percent=CPU_MEMORY_THRESHOLD_PERCENT):
    """
    (Phase 2) Checks CPU memory usage and triggers a reset if threshold is exceeded (with hysteresis).
    """
    # Ensure PromptServer singleton is available
    if server.PromptServer.instance is None:
        return

    # Stability/optimization: Do not trigger during active execution
    if server.PromptServer.instance.prompt_queue.currently_running:
        return

    vm = psutil.virtual_memory()
    current_usage = vm.percent

    if current_usage > threshold_percent:
        # Hysteresis gating
        if current_usage > (_last_cpu_usage_at_reset + CPU_RESET_HYSTERESIS_PERCENT):
            logger.warning(f"[MultiGPU_Memory_Monitor] CPU usage ({current_usage:.1f}%) exceeds threshold ({threshold_percent:.1f}%) and hysteresis.")
            multigpu_memory_log("cpu_monitor", f"trigger:{current_usage:.1f}pct")
            trigger_executor_cache_reset(reason="cpu_threshold_exceeded", force=False)
        else:
            logger.debug(f"[MultiGPU_Memory_Monitor] CPU usage high ({current_usage:.1f}%) but within hysteresis range. Skipping reset.")
            multigpu_memory_log("cpu_monitor", f"skip_hysteresis:{current_usage:.1f}pct")

def get_device_list():
    """
    Enumerate ALL physically available devices that can store torch tensors.
    This includes all device types supported by ComfyUI core.
    Results are cached after first call since devices don't change during runtime.
    
    Returns a comprehensive list of all available devices across all types:
    - CPU (always available)
    - CUDA devices (NVIDIA GPUs)
    - XPU devices (Intel GPUs)
    - NPU devices (Ascend NPUs from Huawei)
    - MLU devices (Cambricon MLUs)
    - MPS device (Apple Metal)
    - DirectML devices (Windows DirectML)
    - CoreX/IXUCA devices
    """
    global _DEVICE_LIST_CACHE
    
    # Return cached result if already populated
    if _DEVICE_LIST_CACHE is not None:
        return _DEVICE_LIST_CACHE
    
    # First time - do the actual detection
    devs = []
    
    # CPU is always physically present and can store tensors
    devs.append("cpu")
    
    # CUDA devices (NVIDIA GPUs)
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devs += [f"cuda:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CUDA device(s)")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] CUDA detection failed: {e}")
    
    # XPU devices (Intel GPUs)
    try:
        # Try to import intel extension first (may be required for XPU support)
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
    try:
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            devs += [f"xpu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} XPU device(s)")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] XPU detection failed: {e}")
    
    # NPU devices (Ascend NPUs from Huawei)
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            device_count = torch.npu.device_count()
            devs += [f"npu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} NPU device(s)")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] NPU detection failed: {e}")
    
    # MLU devices (Cambricon MLUs)
    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            device_count = torch.mlu.device_count()
            devs += [f"mlu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} MLU device(s)")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] MLU detection failed: {e}")
    
    # MPS device (Apple Metal - single device only)
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devs.append("mps")
            logger.debug("[MultiGPU_Device_Utils] Found MPS device")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] MPS detection failed: {e}")
    
    # DirectML devices (Windows DirectML for AMD/Intel/NVIDIA)
    try:
        import torch_directml
        adapter_count = torch_directml.device_count()
        if adapter_count > 0:
            devs += [f"directml:{i}" for i in range(adapter_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {adapter_count} DirectML adapter(s)")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] DirectML detection failed: {e}")
    
    # IXUCA/CoreX devices (special accelerator)
    try:
        if hasattr(torch, "corex"):
            # CoreX typically exposes single device, but check if there's a count method
            if hasattr(torch.corex, "device_count"):
                device_count = torch.corex.device_count()
                devs += [f"corex:{i}" for i in range(device_count)]
                logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CoreX device(s)")
            else:
                devs.append("corex:0")
                logger.debug("[MultiGPU_Device_Utils] Found CoreX device")
    except Exception as e:
        logger.debug(f"[MultiGPU_Device_Utils] CoreX detection failed: {e}")
    
    # Cache the result for future calls
    _DEVICE_LIST_CACHE = devs
    
    # Log only once when initially populated
    logger.debug(f"[MultiGPU_Device_Utils] Device list initialized: {devs}")
    
    return devs


def is_accelerator_available():
    """
    Check if any accelerator device is available.
    Used by patched functions to determine CPU fallback.
    
    Returns True if any GPU/accelerator is available, False otherwise.
    """
    # Check CUDA
    try:
        if torch.cuda.is_available():
            return True
    except:
        pass
    
    # Check XPU (Intel GPU)
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except:
        pass
    
    # Check NPU (Ascend)
    try:
        import torch_npu
        if hasattr(torch, "npu") and torch.npu.is_available():
            return True
    except:
        pass
    
    # Check MLU (Cambricon)
    try:
        import torch_mlu
        if hasattr(torch, "mlu") and torch.mlu.is_available():
            return True
    except:
        pass
    
    # Check MPS (Apple Metal)
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except:
        pass
    
    # Check DirectML
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            return True
    except:
        pass
    
    # Check CoreX/IXUCA
    try:
        if hasattr(torch, "corex"):
            return True
    except:
        pass
    
    return False


def is_device_compatible(device_string):
    """
    Check if a device string represents a valid, available device.
    
    Args:
        device_string: Device identifier like "cuda:0", "cpu", "xpu:1", etc.
    
    Returns:
        True if the device is available, False otherwise.
    """
    available_devices = get_device_list()
    return device_string in available_devices


def get_device_type(device_string):
    """
    Extract the device type from a device string.
    
    Args:
        device_string: Device identifier like "cuda:0", "cpu", "xpu:1", etc.
    
    Returns:
        Device type string (e.g., "cuda", "cpu", "xpu", "npu", "mlu", "mps", "directml", "corex")
    """
    if ":" in device_string:
        return device_string.split(":")[0]
    return device_string


def parse_device_string(device_string):
    """
    Parse a device string into type and index.

    Args:
        device_string: Device identifier like "cuda:0", "cpu", "xpu:1", etc.

    Returns:
        Tuple of (device_type, device_index) where index is None for non-indexed devices
    """
    if ":" in device_string:
        parts = device_string.split(":")
        return parts[0], int(parts[1])
    return device_string, None


def soft_empty_cache_multigpu():
    """
    Replicate ComfyUI's cache clearing but for ALL devices in MultiGPU.
    MultiGPU adaptation of ComfyUI's soft_empty_cache() functionality.
    Uses context managers to ensure the calling thread's device context is restored.
    """
    import gc

    logger.mgpu_mm_log("soft_empty_cache_multigpu: starting GC and multi-device cache clear")
    # Record pre-GC snapshot for general system view
    multigpu_memory_log("general", "pre-soft-empty")

    multigpu_memory_log("general", "pre-gc")
    # Lifecycle status before GC
    log_tracked_modelpatchers_status(tag="pre-gc")
    gc.collect()
    # Lifecycle status after GC
    log_tracked_modelpatchers_status(tag="post-gc")
    multigpu_memory_log("general", "post-gc")
    logger.mgpu_mm_log("soft_empty_cache_multigpu: garbage collection complete")
    # Attempt to release freed heap memory to OS
    try_malloc_trim()

    # Clear cache for ALL devices (not just ComfyUI's single device)
    all_devices = get_device_list()
    logger.mgpu_mm_log(f"soft_empty_cache_multigpu: devices to clear = {all_devices}")
    
    # Check global availability first to avoid unnecessary iteration if backend is missing
    is_cuda_available = hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available()

    for device_str in all_devices:
        if device_str.startswith("cuda:"):
            if is_cuda_available:
                device_idx = int(device_str.split(":")[1])
                # Use context manager for safe switching and automatic restoration
                logger.mgpu_mm_log(f"Clearing CUDA cache on {device_str} (idx={device_idx})")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                with torch.cuda.device(device_idx):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()  # ComfyUI's CUDA optimization
                logger.mgpu_mm_log(f"Cleared CUDA cache (and IPC if available) on {device_str}")
                multigpu_memory_log("general", f"post-empty:{device_str}")

        elif device_str == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                logger.mgpu_mm_log("Clearing MPS cache")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                torch.mps.empty_cache()
                logger.mgpu_mm_log("Cleared MPS cache")
                multigpu_memory_log("general", f"post-empty:{device_str}")

        elif device_str.startswith("xpu:"):
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                logger.mgpu_mm_log(f"Clearing XPU cache on {device_str}")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                torch.xpu.empty_cache()
                logger.mgpu_mm_log(f"Cleared XPU cache on {device_str}")
                multigpu_memory_log("general", f"post-empty:{device_str}")

        elif device_str.startswith("npu:"):
            if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                logger.mgpu_mm_log(f"Clearing NPU cache on {device_str}")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                torch.npu.empty_cache()
                logger.mgpu_mm_log(f"Cleared NPU cache on {device_str}")
                multigpu_memory_log("general", f"post-empty:{device_str}")

        elif device_str.startswith("mlu:"):
            if hasattr(torch, "mlu") and hasattr(torch.mlu, "empty_cache"):
                logger.mgpu_mm_log(f"Clearing MLU cache on {device_str}")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                torch.mlu.empty_cache()
                logger.mgpu_mm_log(f"Cleared MLU cache on {device_str}")
                multigpu_memory_log("general", f"post-empty:{device_str}")

        elif device_str.startswith("corex:"):
            if hasattr(torch, "corex") and hasattr(torch.corex, "empty_cache"):
                logger.mgpu_mm_log(f"Clearing CoreX cache on {device_str}")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                torch.corex.empty_cache()
                logger.mgpu_mm_log(f"Cleared CoreX cache on {device_str}")
                multigpu_memory_log("general", f"post-empty:{device_str}")

    # Record post-GC snapshot for general system view
    multigpu_memory_log("general", "post-soft-empty")



def _bytes_to_gib(b: int) -> float:
    """Convert bytes to GiB as a float."""
    return float(b) / (1024.0 ** 3)


def comfyui_memory_load(tag: str) -> str:
    """
    Returns a single-line, pipe-delimited snapshot of system and device memory usage.

    Format: "tag=<TAG>|cpu=<used_GiB>/<total_GiB>|<device>=<used_GiB>/<total_GiB>|..."
    - CPU values represent system RAM via psutil.
    - Device values represent VRAM via comfy.model_management across all non-CPU devices.
    - Device identifiers use the torch device string from get_device_list() (e.g., 'cuda:0', 'xpu:0', 'mps').
    - Values are in GiB with 2 decimals.
    """
    # CPU RAM
    vm = psutil.virtual_memory()
    cpu_used_gib = _bytes_to_gib(vm.used)
    cpu_total_gib = _bytes_to_gib(vm.total)

    segments = [f"tag={tag}", f"cpu={cpu_used_gib:.2f}/{cpu_total_gib:.2f}"]

    # Enumerate non-CPU devices
    devices = [d for d in get_device_list() if d != "cpu"]

    # Append per-device VRAM used/total
    for dev_str in devices:
        device = torch.device(dev_str)
        total = mm.get_total_memory(device)
        free_info = mm.get_free_memory(device, torch_free_too=True)
        # free_info may be a tuple (system_free, torch_cache_free) or a single value
        if isinstance(free_info, tuple):
            system_free = free_info[0]
        else:
            system_free = free_info
        used = max(0, (total or 0) - (system_free or 0))

        used_gib = _bytes_to_gib(used)
        total_gib = _bytes_to_gib(total or 0)
        if total_gib > 0:
            segments.append(f"{dev_str}={used_gib:.2f}/{total_gib:.2f}")

    return "|".join(segments)


# ==========================================================================================
# Delta-capable memory logging (identifier + tag) with timestamped series
# ==========================================================================================


# Stores the last snapshot per identifier: identifier -> (last_tag, snapshot_map)
# snapshot_map: device_str -> (used_bytes, total_bytes)
_MEM_SNAPSHOT_LAST = {}

# Full chronological series per identifier: identifier -> list[(timestamp, tag, snapshot_map)]
_MEM_SNAPSHOT_SERIES = {}


def _capture_memory_snapshot() -> dict[str, tuple[int, int]]:
    """
    Capture an absolute memory snapshot for CPU and all non-CPU devices.
    Values are returned in bytes (used, total) for each device string key.
    """
    snapshot: dict[str, tuple[int, int]] = {}

    # CPU
    vm = psutil.virtual_memory()
    snapshot["cpu"] = (vm.used, vm.total)

    # Non-CPU devices
    devices = [d for d in get_device_list() if d != "cpu"]
    for dev_str in devices:
        device = torch.device(dev_str)
        total = mm.get_total_memory(device)
        free_info = mm.get_free_memory(device, torch_free_too=True)
        system_free = free_info[0] if isinstance(free_info, tuple) else free_info
        used = max(0, (total or 0) - (system_free or 0))
        snapshot[dev_str] = (used, total or 0)

    return snapshot


def _format_delta_gib(delta_bytes: int) -> str:
    """Format a signed GiB delta with two decimals."""
    gib = _bytes_to_gib(abs(delta_bytes))
    sign = "+" if delta_bytes >= 0 else "-"
    return f"{sign}{gib:.2f}"


def memory_print_summary(log: logging.Logger = logger):
    """
    Print the entire run as absolute actuals with timestamps for each identifier.
    One line per recorded snapshot in insertion order.
    Format:
      YYYY-MM-DDTHH:MM:SS.mmmZ identifier tag | cpu=U/T | cuda:0=U/T | ...
      (GiB values, two decimals)
    """
    from . import logger
    
    # Stable identifier order for readability
    for identifier in sorted(_MEM_SNAPSHOT_SERIES.keys()):
        series = _MEM_SNAPSHOT_SERIES[identifier]
        if not series:
            continue
        logger.mgpu_mm_log(f"=== memory summary: {identifier} ===")
        for ts, tag, snap in series:
            # Build device list (cpu first, then sorted devices)
            parts = []
            # CPU
            cpu_used, cpu_total = snap.get("cpu", (0, 0))
            parts.append(f"cpu={_bytes_to_gib(cpu_used):.2f}/{_bytes_to_gib(cpu_total):.2f}")
            # Non-CPU (sorted)
            devs = sorted([k for k in snap.keys() if k != "cpu"])
            for dev in devs:
                used, total = snap[dev]
                parts.append(f"{dev}={_bytes_to_gib(used):.2f}/{_bytes_to_gib(total):.2f}")
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            logger.mgpu_mm_log(f"{ts_str} {identifier} {tag} | " + " | ".join(parts))


def multigpu_memory_log(identifier: str, tag: str, log: logging.Logger = logger):
    """
    Record a timestamped memory snapshot for the given identifier and tag.
    - INFO: per-device deltas vs. previous snapshot for the same identifier (GiB, signed, no totals).
    - DEBUG: absolute snapshot string via comfyui_memory_load(tag) prefixed by identifier.
    - Special identifier: 'print_summary' will dump the entire series as actuals with timestamps.
    """
    from . import logger as mgpu_logger
    
    if identifier == "print_summary":
        memory_print_summary(log=log)
        return

    # Capture current snapshot and timestamp
    ts = datetime.now(timezone.utc)
    curr = _capture_memory_snapshot()

    # Append to full series
    series = _MEM_SNAPSHOT_SERIES.get(identifier)
    if series is None:
        series = []
        _MEM_SNAPSHOT_SERIES[identifier] = series
    series.append((ts, tag, curr))

    # Compute and log delta vs last
    if identifier in _MEM_SNAPSHOT_LAST:
        prev_tag, prev = _MEM_SNAPSHOT_LAST[identifier]
        # Union of device keys
        keys = set(prev.keys()) | set(curr.keys())
        # Stable order: cpu first, then sorted devices
        ordered = ["cpu"] + sorted([k for k in keys if k != "cpu"])
        parts = []
        for k in ordered:
            p_used, _p_tot = prev.get(k, (0, curr.get(k, (0, 0))[1]))
            c_used, _c_tot = curr.get(k, (0, prev.get(k, (0, 0))[1]))
            delta = c_used - p_used
            parts.append(f"{k}={_format_delta_gib(delta)}")
        logger.mgpu_mm_log(f"{identifier} {tag} - {prev_tag}: " + " | ".join(parts))
    else:
        # Baseline vs zero
        keys = set(curr.keys())
        ordered = ["cpu"] + sorted([k for k in keys if k != "cpu"])
        parts = []
        for k in ordered:
            c_used, _c_tot = curr.get(k, (0, 0))
            parts.append(f"{k}=+{_bytes_to_gib(c_used):.2f}")
        logger.mgpu_mm_log(f"{identifier} {tag} - <baseline>: " + " | ".join(parts))

    # Update last snapshot
    _MEM_SNAPSHOT_LAST[identifier] = (tag, curr)


# ==========================================================================================
# Lifecycle Tracking and Leak Analysis Utilities
# ==========================================================================================

# Track ModelPatcher lifecycle to correlate with CPU RAM trends
if '_MGPU_TRACKED_MODELPATCHERS' not in globals():
    _MGPU_TRACKED_MODELPATCHERS = weakref.WeakSet()

def track_modelpatcher(model_patcher):
    """Registers a ModelPatcher instance for lifecycle tracking."""
    try:
        if isinstance(model_patcher, comfy.model_patcher.ModelPatcher):
            if model_patcher not in _MGPU_TRACKED_MODELPATCHERS:
                _MGPU_TRACKED_MODELPATCHERS.add(model_patcher)
                logger.debug(f"[MultiGPU_Lifecycle] Tracking ModelPatcher {id(model_patcher)} (tracked={len(_MGPU_TRACKED_MODELPATCHERS)})")
    except Exception as e:
        logger.debug(f"[MultiGPU_Lifecycle] track_modelpatcher error: {e}")

def log_tracked_modelpatchers_status(tag="checkpoint"):
    """Logs count and estimated CPU RAM for tracked ModelPatchers."""
    alive_count = len(_MGPU_TRACKED_MODELPATCHERS)
    total_cpu_memory_mb = 0.0
    for patcher in list(_MGPU_TRACKED_MODELPATCHERS):
        try:
            if hasattr(patcher, "model") and patcher.model is not None:
                for param in patcher.model.parameters():
                    if getattr(param, "device", torch.device("cpu")).type == "cpu":
                        total_cpu_memory_mb += (param.nelement() * param.element_size()) / (1024.0 * 1024.0)
        except Exception:
            continue
    logger.warning(f"[MultiGPU_Lifecycle] [{tag}] Tracked ModelPatchers={alive_count}, approx CPU RAM={total_cpu_memory_mb:.2f} MB")

def analyze_cpu_memory_leaks(force=False):
    """Diagnostic: scan referrers of tracked ModelPatchers when memory is high."""
    try:
        vm = psutil.virtual_memory()
        patchers = list(_MGPU_TRACKED_MODELPATCHERS)
        if not force and len(patchers) <= 5 and vm.percent <= 80.0:
            logger.debug(f"[MultiGPU_Leak_Analyzer] Skipping analysis. Patcher count ({len(patchers)}) and memory usage ({vm.percent:.1f}%) normal.")
            return
        logger.warning(f"[MultiGPU_Leak_Analyzer] High pressure (patchers={len(patchers)}, cpu_mem={vm.percent:.1f}%). Inspecting up to 5 referrer sets.")
        for i, patcher in enumerate(patchers[:5]):
                try:
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
                except Exception:
                    logger.warning("[MultiGPU_Leak_Analyzer] Failed to inspect referrers for a patcher.")
    except Exception as e:
        logger.debug(f"[MultiGPU_Leak_Analyzer] analyze error: {e}")

def try_malloc_trim():
    """Attempt to return freed heap memory to OS (Linux/glibc)."""
    try:
        if platform.system() == "Linux":
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                logger.info("[MultiGPU_Memory_Management] malloc_trim(0) begin")
                multigpu_memory_log("mem_mgmt", "pre-malloc-trim")
                res = libc.malloc_trim(0)
                multigpu_memory_log("mem_mgmt", "post-malloc-trim")
                if res == 1:
                    logger.info("[MultiGPU_Memory_Management] malloc_trim(0) released memory")
                else:
                    logger.debug("[MultiGPU_Memory_Management] malloc_trim(0) no release")
    except Exception as e:
        logger.debug(f"[MultiGPU_Memory_Management] malloc_trim error: {e}")

def prune_distorch_stores():
    """Prune stale allocation/settings entries not tied to active models."""
    try:
        multigpu_memory_log("distorch_prune", "start")
        active_hashes_v2 = set()
        active_hashes_v1 = set()
        for lm in getattr(mm, "current_loaded_models", []):
            mp = getattr(lm, "model", None)
            if mp is not None:
                try:
                    h2 = create_safetensor_model_hash(mp, "prune_check_v2")
                    active_hashes_v2.add(h2)
                except Exception:
                    pass
                if create_model_hash is not None:
                    try:
                        h1 = create_model_hash(mp, "prune_check_v1")
                        active_hashes_v1.add(h1)
                    except Exception:
                        pass

        # V1
        if isinstance(model_allocation_store, dict) and active_hashes_v1:
            stale = set(model_allocation_store.keys()) - active_hashes_v1
            if stale:
                logger.info(f"[MultiGPU_Memory_Management] Pruning {len(stale)} DisTorch V1 entries")
                for k in stale:
                    model_allocation_store.pop(k, None)

        # V2
        for store, name in ((safetensor_allocation_store, "allocation"), (safetensor_settings_store, "settings")):
            try:
                if isinstance(store, dict):
                    stale2 = set(store.keys()) - active_hashes_v2
                    if stale2:
                        logger.info(f"[MultiGPU_Memory_Management] Pruning {len(stale2)} V2 {name} entries")
                        for k in stale2:
                            store.pop(k, None)
            except Exception:
                pass
        multigpu_memory_log("distorch_prune", "end")
    except Exception as e:
        logger.debug(f"[MultiGPU_Memory_Management] prune_distorch_stores error: {e}")


# ==========================================================================================
# Model Management Inspection Utilities (End-to-End Tracking)
# ==========================================================================================

def create_model_identifier(model_patcher):
    """Creates a concise, unique identifier for a model patcher based on type and size."""
    if not model_patcher or not model_patcher.model:
        return "N/A (Detached)"

    model = model_patcher.model
    model_type = type(model).__name__

    # Try the fast path first (using size calculated by ModelPatcher)
    try:
        model_size = model_patcher.model_size()
    except Exception:
        model_size = 0

    # If the fast path fails or returns 0, perform a safe deep inspection
    if model_size == 0:
        try:
            # Safely inspect parameters without triggering hooks/loads
            with model_patcher.use_ejected(skip_and_inject_on_exit_only=True):
                 # We must iterate parameters() AND buffers() as both consume memory
                 params = list(model.parameters()) + list(model.buffers())
                 # Use data_ptr to handle potential weight tying/shared tensors correctly
                 seen_tensors = set()
                 for p in params:
                     if p.data_ptr() not in seen_tensors:
                        model_size += p.numel() * p.element_size()
                        seen_tensors.add(p.data_ptr())
        except Exception as e:
            logger.debug(f"[MultiGPU_Inspection] Error during safe size calculation for identifier: {e}")
            return f"{model_type} (ID_Err)"

    # Create a hash based on type and calculated size
    identifier = f"{model_type}_{model_size}"
    model_hash = hashlib.sha256(identifier.encode()).hexdigest()
    return f"{model_type} ({model_hash[:8]})"


def analyze_tensor_locations(model_patcher):
    """
    Analyzes the physical device placement of model tensors (parameters and buffers).
    This provides the Ground Truth location of the data, handling shared weights correctly.
    """
    device_summary = {}
    seen_tensors = set()
    total_memory = 0

    if not model_patcher or not model_patcher.model:
        return {"error": "Model not available"}, 0

    model = model_patcher.model

    # Crucial: Use the ejector to ensure we can access the model weights safely
    # without interfering with injections, hooks, or triggering unintended loads (like in standard LowVRAM mode).
    try:
        with model_patcher.use_ejected(skip_and_inject_on_exit_only=True):
            # Helper to process tensors (parameters or buffers)
            def process_tensor(tensor):
                nonlocal total_memory
                # Use data_ptr() for unique identification of the underlying memory
                if tensor.data_ptr() in seen_tensors:
                    return
                seen_tensors.add(tensor.data_ptr())

                if tensor.numel() > 0:
                    tensor_mem = tensor.numel() * tensor.element_size()
                    total_memory += tensor_mem

                    if hasattr(tensor, 'device'):
                        device = str(tensor.device)
                    else:
                        # Handle cases like NF4 quantization or other custom tensors
                        device = "Unknown/Managed"

                    if device not in device_summary:
                        device_summary[device] = {'tensors': 0, 'memory': 0}

                    device_summary[device]['tensors'] += 1
                    device_summary[device]['memory'] += tensor_mem

            # Iterate over all parameters (weights, biases)
            for param in model.parameters():
                process_tensor(param)

            # Iterate over all buffers (like batch norm running stats)
            for buffer in model.buffers():
                process_tensor(buffer)

    except Exception as e:
        logger.error(f"[MultiGPU_Inspection] Error during tensor location analysis: {e}")
        return {"error": str(e)}, 0

    return device_summary, total_memory


def inspect_model_management_state(context_description=""):
    """
    Provides a detailed, structured overview of the current state of ComfyUI's model management,
    including memory usage across all devices and the status, location, and patching of all loaded models.

    Call this function anywhere in the code to get an immediate snapshot of the system state.
    """

    # Ensure logger configuration (handles calls before full MultiGPU init if needed)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Default to INFO if log level isn't set by main __init__.py
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)

    # We inspect the state without forcing GC or cache clearing, which might alter the state we want to observe.

    logger.info("\n" + "=" * 100)
    logger.info(f"  INSPECTION: ComfyUI Model Management State [Context: {context_description}]")
    logger.info("=" * 100)

    # 1. Device Memory Overview
    # Provides context on available resources across the system.
    logger.info("--- [1] System Device Memory Overview (GB) ---")
    # Sys Free: Memory available to the OS. Torch Alloc: Memory reserved by PyTorch (Active + Cache).
    fmt_mem = "{:<12} | {:>10} | {:>10} | {:>10} | {:>15}"
    logger.info(fmt_mem.format("Device", "Total", "Sys Free", "Used", "Torch Alloc"))
    logger.info("-" * 70)

    all_devices = get_device_list()
    # Sort devices for consistent display (CPU last)
    sorted_devices = sorted(all_devices, key=lambda d: (d == 'cpu', d))

    for dev_str in sorted_devices:
        try:
            device = torch.device(dev_str)

            if dev_str == "cpu":
                vm = psutil.virtual_memory()
                mem_total, mem_free_sys, mem_used = vm.total, vm.available, vm.used
                torch_alloc = 0 # Difficult to track accurately for CPU globally
            else:
                # Use ComfyUI's management functions which account for different backends (CUDA, XPU, etc.)
                mem_total = mm.get_total_memory(device)

                # get_free_memory returns (system_free, torch_cache_free)
                free_info = mm.get_free_memory(device, torch_free_too=True)
                if isinstance(free_info, tuple):
                     mem_free_sys = free_info[0]
                else:
                     mem_free_sys = free_info # Fallback for backends that return single value (like MPS)

                mem_used = mem_total - mem_free_sys

                # Determine Torch Allocation (Reserved memory) - Specific checks for known backends
                torch_alloc = 0
                if device.type == 'cuda' and hasattr(torch.cuda, 'memory_stats'):
                    stats = torch.cuda.memory_stats(device)
                    torch_alloc = stats.get('reserved_bytes.all.current', 0)
                elif device.type == 'xpu' and hasattr(torch, 'xpu') and hasattr(torch.xpu, 'memory_stats'):
                    stats = torch.xpu.memory_stats(device)
                    torch_alloc = stats.get('reserved_bytes.all.current', 0)
                elif device.type == 'npu' and hasattr(torch, 'npu') and hasattr(torch.npu, 'memory_stats'):
                     stats = torch.npu.memory_stats(device)
                     torch_alloc = stats.get('reserved_bytes.all.current', 0)
                elif device.type == 'mlu' and hasattr(torch, 'mlu') and hasattr(torch.mlu, 'memory_stats'):
                     stats = torch.mlu.memory_stats(device)
                     torch_alloc = stats.get('reserved_bytes.all.current', 0)
                # MPS, DirectML, CoreX do not always expose detailed reserved memory stats easily.

            logger.info(fmt_mem.format(
                dev_str,
                f"{mem_total / (1024**3):.2f}",
                f"{mem_free_sys / (1024**3):.2f}",
                f"{mem_used / (1024**3):.2f}",
                f"{torch_alloc / (1024**3):.2f}"
            ))
        except Exception as e:
            logger.debug(f"Could not retrieve memory stats for {dev_str}: {e}")

    logger.info("-" * 70)

    # 2. Loaded Models Inspection (Logical and Physical View)
    # mm.current_loaded_models holds the list of models ComfyUI is managing.
    loaded_models = mm.current_loaded_models
    logger.info(f"\n--- [2] Loaded Models Inspection (Count: {len(loaded_models)}) ---")

    if not loaded_models:
        logger.info("No models currently managed by comfy.model_management.")
        logger.info("=" * 100)
        return

    for i, lm in enumerate(loaded_models):
        logger.info(f"\nModel {i+1}/{len(loaded_models)}:")

        # Check lifecycle status
        mp = lm.model # weakref call to ModelPatcher
        if mp is None:
            # ModelPatcher is gone. Check if the underlying model is still alive (potential leak)
            if lm.is_dead() and lm.real_model() is not None:
                 logger.warning(f"  [!] Status: LEAK DETECTED (Patcher GC'd, but underlying model {lm.real_model().__class__.__name__} persists)")
            else:
                 logger.info(f"  Status: Cleaned Up (Patcher and Model GC'd)")
            continue

        model_id = create_model_identifier(mp)
        logger.info(f"  Identifier: {model_id}")
        logger.info(f"  Status: {'Active (In Use)' if lm.currently_used else 'Idle (Cache)'}")

        # A. Logical View (What ComfyUI intends/tracks)
        logger.info("  [A] Logical View (ComfyUI Tracking):")

        # Devices: Target (Compute) vs Offload (Storage)
        logger.info(f"    Devices: Target={lm.device} | Offload={mp.offload_device} | Current (Model.device)={mp.current_loaded_device()}")

        # Memory Footprint
        mem_total = lm.model_memory()
        mem_loaded = lm.model_loaded_memory()
        mem_offloaded = lm.model_offloaded_memory()
        logger.info(f"    Memory (MB): Total={mem_total/(1024**2):.2f} | Loaded (on Target)={mem_loaded/(1024**2):.2f} | Offloaded={mem_offloaded/(1024**2):.2f}")

        # Management Mode (LowVRAM/DisTorch)
        # model_lowvram indicates if ComfyUI is managing this model partially
        is_lowvram = getattr(mp.model, 'model_lowvram', False)
        lowvram_patches_pending = mp.lowvram_patch_counter()
        logger.info(f"    Mode: {'Partial Load (LowVRAM/DisTorch)' if is_lowvram else 'Full Load'}")
        if is_lowvram:
            # This indicates how many weights are being managed by the partial loading system
            logger.info(f"    Weights Managed by LowVRAM/DisTorch System: {lowvram_patches_pending}")

        # Patching (LoRAs, etc.) - Tracking Attach/Detach
        num_weight_patches = len(mp.patches)
        # Check the UUID applied to the actual weights vs the UUID defined in the patcher
        current_weight_uuid = getattr(mp.model, 'current_weight_patches_uuid', None)
        weights_synced = (mp.patches_uuid == current_weight_uuid) and (current_weight_uuid is not None)

        if num_weight_patches > 0:
            status = 'Applied & Synced' if weights_synced else 'Pending/Mismatch (Re-patch needed)'
            logger.info(f"    Patches: {num_weight_patches} weight patches defined | Status: {status}")
            logger.info(f"    UUIDs: Defined={str(mp.patches_uuid)[:8]}... | Applied={str(current_weight_uuid)[:8] if current_weight_uuid else 'None'}...")

        # B. Physical View (Ground Truth Tensor Locations)
        logger.info("  [B] Physical View (Ground Truth Tensor Locations):")
        device_summary, calculated_total_mem = analyze_tensor_locations(mp)

        if "error" in device_summary:
            logger.error(f"    Analysis Error: {device_summary['error']}")
            continue

        if not device_summary:
            logger.info("    No tensors found (e.g., fully offloaded CLIP or utility object).")
        else:
            # Sort devices (CPU last)
            sorted_devices = sorted(device_summary.keys(), key=lambda d: (d.startswith("cpu"), d))
            fmt_loc = "    {:<15} | Tensors: {:>6} | Memory (MB): {:>10.2f} | Percent: {:>6.1f}%"
            for device in sorted_devices:
                data = device_summary[device]
                percent = (data['memory'] / calculated_total_mem) * 100 if calculated_total_mem > 0 else 0
                logger.info(fmt_loc.format(device, data['tensors'], data['memory']/(1024**2), percent))

            # Verification Check
            if abs(calculated_total_mem - mem_total) > (1024*1024): # Allow 1MB difference
                logger.warning(f"    [!] Verification WARNING: Physical memory ({calculated_total_mem/(1024**2):.2f}MB) differs from logical memory ({mem_total/(1024**2):.2f}MB).")

        logger.info("-" * 100)

    logger.info("End of Inspection")
    logger.info("=" * 100)
