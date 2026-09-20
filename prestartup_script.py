import os
import logging

# --- CONFIGURATION PARAMETER ---
# Set this to True to see logs, or False to hide them.
# It also checks if an environment variable 'AIMDO_DEBUG' is set to 'true'.
ENABLE_LOGGING = os.environ.get("AIMDO_DEBUG", "true").lower() == "false"

log = logging.getLogger("MultiGPU.AIMDO")

def _debug_log(*args, **kwargs):
    """Helper function to print only if ENABLE_LOGGING is True."""
    if ENABLE_LOGGING:
        print(*args, **kwargs)

def _safe_hex(ptr):
    try:
        return hex(ptr) if ptr else None
    except Exception:
        return str(ptr)

def _patch_aimdo_init_for_multigpu():
    try:
        import torch
        import comfy_aimdo.control as control
    except Exception as e:
        _debug_log("[MultiGPU.AIMDO] prestartup: aimdo/torch import failed:", e)
        return

    if getattr(control, "_multigpu_patch_applied", False):
        return
    control._multigpu_patch_applied = True

    orig_init = control.init
    orig_init_device = control.init_device
    orig_init_devices = control.init_devices

    def init_devices(device_ids):
        ok = orig_init_devices(device_ids)
        _debug_log(f"[MultiGPU.AIMDO] init_devices({list(device_ids)}) -> {ok}")
        if getattr(control, "lib", None) is not None:
            for d in device_ids:
                try:
                    ptr = control.lib.get_devctx(int(d))
                except Exception:
                    ptr = None
                _debug_log(f"[MultiGPU.AIMDO]   devctx cuda:{d} = {_safe_hex(ptr)}")
        return ok

    def init_device(device_id: int):
        try:
            if control.lib is None:
                _debug_log("[MultiGPU.AIMDO] control.lib is None -> calling control.init()")
                orig_init()

            if not torch.cuda.is_available():
                _debug_log("[MultiGPU.AIMDO] torch.cuda.is_available() is False -> falling back to original init_device")
                return orig_init_device(device_id)

            count = torch.cuda.device_count()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            _debug_log(f"[MultiGPU.AIMDO] init_device({device_id}) called. cuda_count={count} CUDA_VISIBLE_DEVICES={visible!r}")

            if count <= 1:
                return orig_init_device(device_id)

            if getattr(control, "devctxs", None):
                present = []
                if control.lib is not None:
                    for d in range(count):
                        try:
                            if control.lib.get_devctx(int(d)):
                                present.append(d)
                        except Exception:
                            pass
                _debug_log(f"[MultiGPU.AIMDO] AIMDO already initialized. present_devctx={present} devctxs_len={len(control.devctxs)}")
                
                if len(present) == count:
                    return True
                
                _debug_log("[MultiGPU.AIMDO] WARNING: not all devices initialized. Please fully restart ComfyUI now that the patch is installed.")
                return True

            devs = list(range(count))
            return init_devices(devs)

        except Exception as e:
            _debug_log("[MultiGPU.AIMDO] ERROR in patched init_device:", e)
            return orig_init_device(device_id)

    # Apply monkeypatches
    control.init_devices = init_devices
    control.init_device = init_device

    _debug_log("[MultiGPU.AIMDO] Patched comfy_aimdo.control.init_device to initialize ALL visible GPUs.")

_patch_aimdo_init_for_multigpu()