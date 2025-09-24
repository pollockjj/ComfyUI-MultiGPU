"""
Device detection, management, and inspection utilities for ComfyUI-MultiGPU.
Single source of truth for all device enumeration, compatibility checks, and VRAM management.
Handles all device types supported by ComfyUI core.
"""

import torch
import logging
import hashlib
import psutil
import comfy.model_management as mm
import gc

logger = logging.getLogger("MultiGPU")

# Module-level cache for device list (populated once on first call)
_DEVICE_LIST_CACHE = None

# ==========================================================================================
# Device Detection and Management
# ==========================================================================================

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
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devs += [f"cuda:{i}" for i in range(device_count)]
        logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CUDA device(s)")
    
    # XPU devices (Intel GPUs)
    try:
        # Try to import intel extension first (may be required for XPU support)
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
    
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        devs += [f"xpu:{i}" for i in range(device_count)]
        logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} XPU device(s)")
    
    # NPU devices (Ascend NPUs from Huawei)
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            device_count = torch.npu.device_count()
            devs += [f"npu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} NPU device(s)")
    except ImportError:
        pass
    
    # MLU devices (Cambricon MLUs)
    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            device_count = torch.mlu.device_count()
            devs += [f"mlu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} MLU device(s)")
    except ImportError:
        pass
    
    # MPS device (Apple Metal - single device only)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
        logger.debug("[MultiGPU_Device_Utils] Found MPS device")
    
    # DirectML devices (Windows DirectML for AMD/Intel/NVIDIA)
    try:
        import torch_directml
        adapter_count = torch_directml.device_count()
        if adapter_count > 0:
            devs += [f"directml:{i}" for i in range(adapter_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {adapter_count} DirectML adapter(s)")
    except ImportError:
        pass
    
    # IXUCA/CoreX devices (special accelerator)
    try:
        if hasattr(torch, "corex"):
            if hasattr(torch.corex, "device_count"):
                device_count = torch.corex.device_count()
                devs += [f"corex:{i}" for i in range(device_count)]
                logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CoreX device(s)")
            else:
                devs.append("corex:0")
                logger.debug("[MultiGPU_Device_Utils] Found CoreX device")
    except ImportError:
        pass
    
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
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return True
    
    # Check XPU (Intel GPU)
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        return True
    
    # Check NPU (Ascend)
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            return True
    except ImportError:
        pass
    
    # Check MLU (Cambricon)
    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            return True
    except ImportError:
        pass
    
    # Check MPS (Apple Metal)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True
    
    # Check DirectML
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            return True
    except ImportError:
        pass
    
    # Check CoreX/IXUCA
    if hasattr(torch, "corex"):
        return True
    
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

# ==========================================================================================
# VRAM Management (Multi-device cache clearing)
# ==========================================================================================

def soft_empty_cache_multigpu():
    """
    Replicate ComfyUI's cache clearing but for ALL devices in MultiGPU.
    Uses context managers to ensure the calling thread's device context is restored.
    """
    # Import model management functions
    from .model_management_mgpu import multigpu_memory_log, log_tracked_modelpatchers_status, try_malloc_trim
    
    logger.mgpu_mm_log("soft_empty_cache_multigpu: starting GC and multi-device cache clear")
    multigpu_memory_log("general", "pre-soft-empty")

    multigpu_memory_log("general", "pre-gc")
    log_tracked_modelpatchers_status(tag="pre-gc")
    gc.collect()
    log_tracked_modelpatchers_status(tag="post-gc")
    multigpu_memory_log("general", "post-gc")
    logger.mgpu_mm_log("soft_empty_cache_multigpu: garbage collection complete")

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
                logger.mgpu_mm_log(f"Clearing CUDA cache on {device_str} (idx={device_idx})")
                multigpu_memory_log("general", f"pre-empty:{device_str}")
                with torch.cuda.device(device_idx):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
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

    multigpu_memory_log("general", "post-soft-empty")

# ==========================================================================================
# Memory Inspection Utilities
# ==========================================================================================

def comfyui_memory_load(tag):
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
    cpu_used_gib = vm.used / (1024.0 ** 3)
    cpu_total_gib = vm.total / (1024.0 ** 3)

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

        used_gib = used / (1024.0 ** 3)
        total_gib = (total or 0) / (1024.0 ** 3)
        if total_gib > 0:
            segments.append(f"{dev_str}={used_gib:.2f}/{total_gib:.2f}")

    return "|".join(segments)
