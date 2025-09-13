"""
Device detection and management utilities for ComfyUI-MultiGPU.
Single source of truth for all device enumeration and compatibility checks.
Handles all device types supported by ComfyUI core.
"""

import torch
import logging

logger = logging.getLogger("MultiGPU")

# Module-level cache for device list (populated once on first call)
_DEVICE_LIST_CACHE = None

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
    logger.info(f"[MultiGPU_Device_Utils] Device list initialized: {devs}")
    
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


def get_device_type(device_string):
    """
    Extract the device type from a device string.
    """
    if ":" in device_string:
        return device_string.split(":")[0]
    return device_string

def soft_empty_cache_multigpu(logger):
    """
    Replicate ComfyUI's cache clearing but for ALL devices in MultiGPU.
    MultiGPU adaptation of ComfyUI's soft_empty_cache() functionality.
    """

    import gc

    logger.info("[MultiGPU_Device_Utils] Initiating cache clearing across all available devices (Patched mm.soft_empty_cache).")

    # Python GC
    gc.collect()
    logger.debug("[MultiGPU_Device_Utils] Performed Python garbage collection.")

    # Store the current active devices for architectures that require context switching, to restore later
    original_devices = {}

    # Helper to get current device safely
    def get_current_device(device_type_module):
        try:
            if device_type_module.is_available() and hasattr(device_type_module, "current_device"):
                return device_type_module.current_device()
        except Exception as e:
            logger.debug(f"[MultiGPU_Device_Utils] Could not get current device for module {getattr(device_type_module, '__name__', 'unknown')}: {e}")
        return None

    if hasattr(torch, "cuda"):
        original_devices["cuda"] = get_current_device(torch.cuda)
    if hasattr(torch, "xpu"):
        original_devices["xpu"] = get_current_device(torch.xpu)
    if hasattr(torch, "npu"):
        original_devices["npu"] = get_current_device(torch.npu)
    if hasattr(torch, "mlu"):
        original_devices["mlu"] = get_current_device(torch.mlu)

    all_devices = get_device_list()

    for device_str in all_devices:
        try:
            device_type = get_device_type(device_str)

            if device_type == "cuda" and hasattr(torch, "cuda"):
                device_idx = int(device_str.split(":")[1])
                torch.cuda.set_device(device_idx)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.debug(f"[MultiGPU_Device_Utils] Cleared cache + IPC for {device_str}")

            elif device_type == "xpu" and hasattr(torch, "xpu"):
                device_idx = int(device_str.split(":")[1])
                torch.xpu.set_device(device_idx)
                torch.xpu.empty_cache()
                logger.debug(f"[MultiGPU_Device_Utils] Cleared cache for {device_str}")

            elif device_type == "npu" and hasattr(torch, "npu"):
                device_idx = int(device_str.split(":")[1])
                torch.npu.set_device(device_idx)
                torch.npu.empty_cache()
                logger.debug(f"[MultiGPU_Device_Utils] Cleared cache for {device_str}")

            elif device_type == "mlu" and hasattr(torch, "mlu"):
                device_idx = int(device_str.split(":")[1])
                torch.mlu.set_device(device_idx)
                torch.mlu.empty_cache()
                logger.debug(f"[MultiGPU_Device_Utils] Cleared cache for {device_str}")

            elif device_type == "mps":
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    logger.debug("[MultiGPU_Device_Utils] Cleared cache for MPS")

            elif device_type == "corex":
                 if hasattr(torch, "corex") and hasattr(torch.corex, "empty_cache"):
                    torch.corex.empty_cache()
                    logger.debug(f"[MultiGPU_Device_Utils] Cleared cache for {device_str}")

        except Exception as e:
            logger.warning(f"[MultiGPU_Device_Utils] Failed to clear cache for {device_str}. Error: {e}")

    # Restore the original active devices
    # Helper to restore device safely
    def restore_device(device_type_str, device_type_module, original_device_idx):
        if original_device_idx is not None:
            try:
                if device_type_module.current_device() != original_device_idx:
                    device_type_module.set_device(original_device_idx)
                    logger.debug(f"[MultiGPU_Device_Utils] Restored active {device_type_str} device to index {original_device_idx}")
            except Exception as e:
                logger.warning(f"[MultiGPU_Device_Utils] Failed to restore original {device_type_str} device index {original_device_idx}. Error: {e}")

    if hasattr(torch, "cuda"):
        restore_device("cuda", torch.cuda, original_devices.get("cuda"))
    if hasattr(torch, "xpu"):
        restore_device("xpu", torch.xpu, original_devices.get("xpu"))
    if hasattr(torch, "npu"):
        restore_device("npu", torch.npu, original_devices.get("npu"))
    if hasattr(torch, "mlu"):
        restore_device("mlu", torch.mlu, original_devices.get("mlu"))
