import torch
import logging
import hashlib
import psutil
import comfy.model_management as mm
import gc

logger = logging.getLogger("MultiGPU")

_DEVICE_LIST_CACHE = None

def get_device_list():
    """
    Enumerate ALL physically available devices that can store torch tensors.
    This includes all device types supported by ComfyUI core.
    Results are cached after first call since devices don't change during runtime.
    
    Returns a comprehensive list of all available devices across all types:
    - CPU (always available)
    - CUDA devices (NVIDIA GPUs + AMD w/ ROCm GPUs)
    - XPU devices (Intel GPUs)
    - NPU devices (Ascend NPUs from Huawei)
    - MLU devices (Cambricon MLUs)
    - MPS device (Apple Metal)
    - DirectML devices (Windows DirectML)
    - CoreX/IXUCA devices
    """
    global _DEVICE_LIST_CACHE
    
    if _DEVICE_LIST_CACHE is not None:
        return _DEVICE_LIST_CACHE
    
    devs = []
    
    devs.append("cpu")
    
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devs += [f"cuda:{i}" for i in range(device_count)]
        logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} CUDA device(s)")
    
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
    
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        devs += [f"xpu:{i}" for i in range(device_count)]
        logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} XPU device(s)")
    
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            device_count = torch.npu.device_count()
            devs += [f"npu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} NPU device(s)")
    except ImportError:
        pass
    
    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            device_count = torch.mlu.device_count()
            devs += [f"mlu:{i}" for i in range(device_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {device_count} MLU device(s)")
    except ImportError:
        pass
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
        logger.debug("[MultiGPU_Device_Utils] Found MPS device")
    
    try:
        import torch_directml
        adapter_count = torch_directml.device_count()
        if adapter_count > 0:
            devs += [f"directml:{i}" for i in range(adapter_count)]
            logger.debug(f"[MultiGPU_Device_Utils] Found {adapter_count} DirectML adapter(s)")
    except ImportError:
        pass
    
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
    
    _DEVICE_LIST_CACHE = devs
    
    logger.debug(f"[MultiGPU_Device_Utils] Device list initialized: {devs}")
    
    return devs

def is_accelerator_available():
    """Check if any GPU or accelerator device is available including CUDA, XPU, NPU, MLU, MPS, DirectML, or CoreX."""
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return True
    
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        return True
    
    try:
        import torch_npu
        if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            return True
    except ImportError:
        pass

    try:
        import torch_mlu
        if hasattr(torch, "mlu") and hasattr(torch.mlu, "is_available") and torch.mlu.is_available():
            return True
    except ImportError:
        pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True

    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            return True
    except ImportError:
        pass

    if hasattr(torch, "corex"):
        return True
    
    return False

def is_device_compatible(device_string):
    """Check if a device string represents a valid available device."""
    available_devices = get_device_list()
    return device_string in available_devices

def get_device_type(device_string):
    """Extract device type from device string (e.g. 'cuda' from 'cuda:0')."""
    if ":" in device_string:
        return device_string.split(":")[0]
    return device_string

def parse_device_string(device_string):
    """Parse device string into (device_type, device_index) tuple."""
    if ":" in device_string:
        parts = device_string.split(":")
        return parts[0], int(parts[1])
    return device_string, None

def soft_empty_cache_multigpu():
    """Clear allocator caches across all devices using context managers to preserve calling thread device context."""
    from .model_management_mgpu import multigpu_memory_log
    
    logger.mgpu_mm_log("soft_empty_cache_multigpu: starting GC and multi-device cache clear")

    gc.collect()

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
# Comprehensive Memory Management (VRAM + CPU + Store Pruning)
# ==========================================================================================

logger.info("[MultiGPU Core Patching] Patching mm.soft_empty_cache for Comprehensive Memory Management (VRAM + CPU + Store Pruning)")

original_soft_empty_cache = mm.soft_empty_cache

def soft_empty_cache_distorch2_patched(force=False):
    """Patched mm.soft_empty_cache managing VRAM across all devices, CPU RAM with adaptive thresholding, and DisTorch store pruning."""
    from .model_management_mgpu import multigpu_memory_log, check_cpu_memory_threshold, trigger_executor_cache_reset
    
    is_distorch_active = False

    for i, lm in enumerate(mm.current_loaded_models):
        mp = lm.model
        if mp is not None:
            inner_model = mp.model
            
            if hasattr(inner_model, '_distorch_v2_meta'):
                is_distorch_active = True
                break

    # Phase 2: adaptive CPU memory management
    check_cpu_memory_threshold()

    # VRAM allocator management
    if is_distorch_active:
        logger.mgpu_mm_log("DisTorch2 active: clearing allocator caches on all devices (VRAM)")
        soft_empty_cache_multigpu()
    else:
        original_soft_empty_cache(force)
        # Optional: return CPU heap to OS (not part of Comfy Core)

    # Phase 1/3: forced executor reset mirrors ComfyUI 'Free memory' semantics
    if force:
        logger.mgpu_mm_log("Force flag active: triggering executor cache reset (CPU)")
        trigger_executor_cache_reset(reason="forced_soft_empty", force=True)

mm.soft_empty_cache = soft_empty_cache_distorch2_patched

# ==========================================================================================
# Memory Inspection Utilities
# ==========================================================================================

def comfyui_memory_load(tag):
    """Return single-line pipe-delimited snapshot of system and device memory usage in GiB."""
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
