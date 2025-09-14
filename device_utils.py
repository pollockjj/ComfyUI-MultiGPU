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


def soft_empty_cache_multigpu(logger):
    """
    Replicate ComfyUI's cache clearing but for ALL devices in MultiGPU.
    MultiGPU adaptation of ComfyUI's soft_empty_cache() functionality.
    """
    import gc

    logger.info("[MultiGPU_Device_Utils] Preparing devices for optimized safetensor loading")

    # Python GC (same as all implementations)
    gc.collect()
    logger.debug("[MultiGPU_Device_Utils] Performed garbage collection before safetensor loading")

    # Clear cache for ALL devices (not just ComfyUI's single device)
    all_devices = get_device_list()

    for device_str in all_devices:
        if device_str.startswith("cuda:"):
            device_idx = int(device_str.split(":")[1])
            torch.cuda.set_device(device_idx)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # ComfyUI's CUDA optimization
            logger.debug(f"[MultiGPU_Device_Utils] Cleared cache + IPC for {device_str}")
        elif device_str == "mps":
            torch.mps.empty_cache()
            logger.debug("[MultiGPU_Device_Utils] Cleared cache for MPS")
        elif device_str.startswith("xpu:"):
            torch.xpu.empty_cache()
            logger.debug("[MultiGPU_Device_Utils] Cleared cache for Intel XPU")
        elif device_str.startswith("npu:"):
            torch.npu.empty_cache()
            logger.debug("[MultiGPU_Device_Utils] Cleared cache for Ascend NPU")
        elif device_str.startswith("mlu:"):
            torch.mlu.empty_cache()
            logger.debug("[MultiGPU_Device_Utils] Cleared cache for Cambricon MLU")
        elif device_str.startswith("corex:"):
            torch.corex.empty_cache()  # Hypothetical based on ComfyUI's ixuca support
            logger.debug("[MultiGPU_Device_Utils] Cleared cache for CoreX")


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
