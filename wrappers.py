"""
ComfyUI-MultiGPU Wrapper Functions
All node override/wrapper generation functions consolidated in one location
"""

import copy
import hashlib
import logging
from .device_utils import get_device_list

logger = logging.getLogger("MultiGPU")


# ============================================================================
# DISTORCH V2 SAFETENSOR WRAPPERS (DisTorch2 for .safetensors and .gguf)
# ============================================================================

def _create_distorch_safetensor_v2_override(cls, device_param_name, device_setter_func, apply_device_kwarg_workaround, eject_models_default=True):
    """Internal factory function creating DisTorch2 override class with parameterized device selection behavior."""
    from .distorch_2 import register_patched_safetensor_modelpatcher
    from .model_management_mgpu import force_full_system_cleanup
    
    class NodeOverrideDisTorchSafetensorV2(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]

            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"][device_param_name] = (devices, {"default": default_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1})
            inputs["optional"]["donor_device"] = (devices, {"default": "cpu"})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            inputs["optional"]["eject_models"] = ("BOOLEAN", {"default": eject_models_default})
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch2)"

        @classmethod
        def IS_CHANGED(s, *args, virtual_vram_gb=4.0, donor_device="cpu",
                       expert_mode_allocations="", eject_models=eject_models_default, **kwargs):
            device_value = kwargs.get(device_param_name)
            settings_str = f"{device_value}{virtual_vram_gb}{donor_device}{expert_mode_allocations}{eject_models}"
            current_hash = hashlib.sha256(settings_str.encode()).hexdigest()

            if not hasattr(cls, '_last_hash'):
                cls._last_hash = current_hash
                logger.mgpu_mm_log(f"IS_CHANGED first call: {current_hash[:8]}")
            elif cls._last_hash != current_hash:
                cls._last_hash = current_hash
                logger.mgpu_mm_log(f"IS_CHANGED CHANGED: {current_hash[:8]} ← settings changed")
            return current_hash

        def override(self, *args, virtual_vram_gb=4.0, donor_device="cpu",
                     expert_mode_allocations="", eject_models=eject_models_default, **kwargs):

            device_value = kwargs.get(device_param_name)

            # Capture the current device at runtime so we can restore it later
            from . import get_current_device, get_current_text_encoder_device
            original_device = get_current_device() if device_param_name == "compute_device" else get_current_text_encoder_device()

            import comfy.model_management as mm

            if eject_models:
                logger.mgpu_mm_log(f"[EJECT_MODELS_SETUP] eject_models=True - marking all loaded models for eviction, device target: {device_value}")
                ejection_count = 0
                for i, lm in enumerate(mm.current_loaded_models):
                    # Set _mgpu_unload_distorch_model=True on all models to force Comfy Core eviction
                    model_name = type(getattr(lm.model, 'model', lm.model)).__name__ if lm.model else 'Unknown'

                    if hasattr(lm.model, 'model') and lm.model.model is not None:
                        lm.model.model._mgpu_unload_distorch_model = True
                        logger.mgpu_mm_log(f"[EJECT_MARKED] Model {i}: {model_name} (id=0x{id(lm):x}) → marked for eviction")
                        ejection_count += 1
                    elif lm.model is not None:
                        lm.model._mgpu_unload_distorch_model = True
                        logger.mgpu_mm_log(f"[EJECT_MARKED] Model {i}: {model_name} (direct patcher) → marked for eviction")
                        ejection_count += 1

                logger.mgpu_mm_log(f"[EJECT_MODELS_SETUP_COMPLETE] Marked {ejection_count} models for Comfy Core eviction during load_models_gpu")
            else:
                logger.mgpu_mm_log(f"[EJECT_MODELS_SETUP] eject_models=False - loading without eviction")

            if device_value is not None:
                device_setter_func(device_value)

            # Strip MultiGPU-specific parameters before calling original function (REMOVE eject_models, eject_models and virtual_vram_gb since we handle them above)
            clean_kwargs = {k: v for k, v in kwargs.items()
                           if k not in [device_param_name, 'virtual_vram_gb',
                                        'donor_device', 'expert_mode_allocations',
                                        'eject_models']}
            
            if apply_device_kwarg_workaround:
                clean_kwargs['device'] = 'default'

            register_patched_safetensor_modelpatcher()

            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{device_value};{virtual_vram_gb};{donor_device}"
            elif expert_mode_allocations:
                vram_string = device_value

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **clean_kwargs)
            
            model_to_check = None
            if hasattr(out[0], 'model'):
                model_to_check = out[0]
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_to_check = out[0].patcher

            if model_to_check and full_allocation:
                inner_model = model_to_check.model
                inner_model._distorch_v2_meta = {"full_allocation": full_allocation}

            logger.info(f"[MultiGPU DisTorch V2] Full allocation string: {full_allocation}")

            try:
                return out
            finally:
                # Restore the device that was in use when the override started
                device_setter_func(original_device)

    return NodeOverrideDisTorchSafetensorV2


def override_class_with_distorch_safetensor_v2(cls):
    """DisTorch 2.0 wrapper for safetensor UNet/VAE models"""
    from . import set_current_device
    return _create_distorch_safetensor_v2_override(
        cls,
        device_param_name="compute_device",
        device_setter_func=set_current_device,
        apply_device_kwarg_workaround=False
    )


def override_class_with_distorch_safetensor_v2_clip(cls):
    """DisTorch 2.0 wrapper for safetensor CLIP models (with device kwarg workaround)"""
    from . import set_current_text_encoder_device
    return _create_distorch_safetensor_v2_override(
        cls,
        device_param_name="device",
        device_setter_func=set_current_text_encoder_device,
        apply_device_kwarg_workaround=True,
        eject_models_default=False  # CLIP defaults to False
    )


def override_class_with_distorch_safetensor_v2_clip_no_device(cls):
    """DisTorch 2.0 wrapper for safetensor Triple/Quad CLIP models (no device kwarg workaround)"""
    from . import set_current_text_encoder_device
    return _create_distorch_safetensor_v2_override(
        cls,
        device_param_name="device",
        device_setter_func=set_current_text_encoder_device,
        apply_device_kwarg_workaround=False,
        eject_models_default=False  # CLIP defaults to False
    )


# ============================================================================
# DISTORCH V1 LEGACY WRAPPERS (Rewritten to call V2 backend)
# ============================================================================

def override_class_with_distorch_gguf(cls):
    """DisTorch V1 Legacy wrapper - maintains V1 UI but calls V2 backend"""
    from . import set_current_device, get_current_device
    from .distorch_2 import register_patched_safetensor_modelpatcher
    
    class NodeOverrideDisTorchGGUFLegacy(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.1})
            inputs["optional"]["use_other_vram"] = ("BOOLEAN", {"default": False})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            return inputs

        CATEGORY = "multigpu/legacy"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (Legacy)"

        def override(self, *args, device=None, expert_mode_allocations="", use_other_vram=False, virtual_vram_gb=0.0, **kwargs):
            # Capture and restore the current device to avoid leaking global state
            original_device = get_current_device()
            if device is not None:
                set_current_device(device)
            
            # Strip MultiGPU-specific parameters before calling original function
            clean_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['device', 'virtual_vram_gb', 'use_other_vram', 
                                        'expert_mode_allocations']}
            
            register_patched_safetensor_modelpatcher()
            
            vram_string = ""
            if virtual_vram_gb > 0:
                if use_other_vram:
                    available_devices = [d for d in get_device_list() if d != "cpu"]
                    other_devices = [d for d in available_devices if d != device]
                    other_devices.sort(key=lambda x: int(x.split(':')[1] if ':' in x else x[-1]), reverse=False)
                    device_string = ','.join(other_devices + ['cpu'])
                    vram_string = f"{device};{virtual_vram_gb};{device_string}"
                else:
                    vram_string = f"{device};{virtual_vram_gb};cpu"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **clean_kwargs)

            model_to_check = None
            if hasattr(out[0], 'model'):
                model_to_check = out[0]
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_to_check = out[0].patcher
            
            if model_to_check and full_allocation:
                inner_model = model_to_check.model
                inner_model._distorch_v2_meta = {"full_allocation": full_allocation}
            try:
                return out
            finally:
                set_current_device(original_device)

    return NodeOverrideDisTorchGGUFLegacy


def override_class_with_distorch_gguf_v2(cls):
    """DisTorch V2 wrapper for GGUF models"""
    from . import set_current_device, get_current_device
    from .distorch_2 import register_patched_safetensor_modelpatcher
    
    class NodeOverrideDisTorchGGUFv2(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            compute_device = devices[1] if len(devices) > 1 else devices[0]
            
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["compute_device"] = (devices, {"default": compute_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1})
            inputs["optional"]["donor_device"] = (devices, {"default": "cpu"})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            return inputs

        CATEGORY = "multigpu/distorch_2"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch2)"

        def override(self, *args, compute_device=None, virtual_vram_gb=4.0, donor_device="cpu", expert_mode_allocations="", **kwargs):
            original_device = get_current_device()
            if compute_device is not None:
                set_current_device(compute_device)
            
            # Strip MultiGPU-specific parameters before calling original function
            clean_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['compute_device', 'virtual_vram_gb', 
                                        'donor_device', 'expert_mode_allocations']}
            
            register_patched_safetensor_modelpatcher()
            
            vram_string = ""
            if virtual_vram_gb > 0:
                vram_string = f"{compute_device};{virtual_vram_gb};{donor_device}"
            elif expert_mode_allocations:
                vram_string = compute_device

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            logger.info(f"[MultiGPU DisTorch V2] Full allocation string: {full_allocation}")
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **clean_kwargs)
            
            model_to_check = None
            if hasattr(out[0], 'model'):
                model_to_check = out[0]
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_to_check = out[0].patcher
            
            if model_to_check and full_allocation:
                inner_model = model_to_check.model
                inner_model._distorch_v2_meta = {"full_allocation": full_allocation}
            try:
                return out
            finally:
                set_current_device(original_device)

    return NodeOverrideDisTorchGGUFv2


def override_class_with_distorch_clip(cls):
    """DisTorch V1 wrapper for CLIP models - calls V2 backend"""
    from . import set_current_text_encoder_device, get_current_text_encoder_device
    from .distorch_2 import register_patched_safetensor_modelpatcher
    
    class NodeOverrideDisTorchClip(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.1})
            inputs["optional"]["use_other_vram"] = ("BOOLEAN", {"default": False})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch)"

        def override(self, *args, device=None, expert_mode_allocations="", use_other_vram=False, virtual_vram_gb=0.0, **kwargs):
            original_text_device = get_current_text_encoder_device()
            if device is not None:
                set_current_text_encoder_device(device)
            
            # Strip MultiGPU-specific parameters before calling original function
            clean_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['device', 'virtual_vram_gb', 'use_other_vram', 
                                        'expert_mode_allocations']}
            
            register_patched_safetensor_modelpatcher()
            
            vram_string = ""
            if virtual_vram_gb > 0:
                if use_other_vram:
                    available_devices = [d for d in get_device_list() if d != "cpu"]
                    other_devices = [d for d in available_devices if d != device]
                    other_devices.sort(key=lambda x: int(x.split(':')[1] if ':' in x else x[-1]), reverse=False)
                    device_string = ','.join(other_devices + ['cpu'])
                    vram_string = f"{device};{virtual_vram_gb};{device_string}"
                else:
                    vram_string = f"{device};{virtual_vram_gb};cpu"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **clean_kwargs)
            
            model_to_check = None
            if hasattr(out[0], 'model'):
                model_to_check = out[0]
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_to_check = out[0].patcher
            
            if model_to_check and full_allocation:
                inner_model = model_to_check.model
                inner_model._distorch_v2_meta = {"full_allocation": full_allocation}
            try:
                return out
            finally:
                set_current_text_encoder_device(original_text_device)

    return NodeOverrideDisTorchClip


def override_class_with_distorch_clip_no_device(cls):
    """DisTorch V1 wrapper for Triple/Quad CLIP models - calls V2 backend"""
    from . import set_current_text_encoder_device, get_current_text_encoder_device
    from .distorch_2 import register_patched_safetensor_modelpatcher
    
    class NodeOverrideDisTorchClipNoDevice(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["virtual_vram_gb"] = ("FLOAT", {"default": 4.0, "min": 0.0, "max": 24.0, "step": 0.1})
            inputs["optional"]["use_other_vram"] = ("BOOLEAN", {"default": False})
            inputs["optional"]["expert_mode_allocations"] = ("STRING", {"multiline": False, "default": ""})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"
        TITLE = f"{cls.TITLE if hasattr(cls, 'TITLE') else cls.__name__} (DisTorch)"

        def override(self, *args, device=None, expert_mode_allocations="", use_other_vram=False, virtual_vram_gb=0.0, **kwargs):
            original_text_device = get_current_text_encoder_device()
            if device is not None:
                set_current_text_encoder_device(device)
            
            # Strip MultiGPU-specific parameters before calling original function
            clean_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['device', 'virtual_vram_gb', 'use_other_vram', 
                                        'expert_mode_allocations']}
            
            register_patched_safetensor_modelpatcher()
            
            vram_string = ""
            if virtual_vram_gb > 0:
                if use_other_vram:
                    available_devices = [d for d in get_device_list() if d != "cpu"]
                    other_devices = [d for d in available_devices if d != device]
                    other_devices.sort(key=lambda x: int(x.split(':')[1] if ':' in x else x[-1]), reverse=False)
                    device_string = ','.join(other_devices + ['cpu'])
                    vram_string = f"{device};{virtual_vram_gb};{device_string}"
                else:
                    vram_string = f"{device};{virtual_vram_gb};cpu"

            full_allocation = f"{expert_mode_allocations}#{vram_string}" if expert_mode_allocations or vram_string else ""
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **clean_kwargs)
            
            model_to_check = None
            if hasattr(out[0], 'model'):
                model_to_check = out[0]
            elif hasattr(out[0], 'patcher') and hasattr(out[0].patcher, 'model'):
                model_to_check = out[0].patcher
            
            if model_to_check and full_allocation:
                inner_model = model_to_check.model
                inner_model._distorch_v2_meta = {"full_allocation": full_allocation}
            try:
                return out
            finally:
                set_current_text_encoder_device(original_text_device)

    return NodeOverrideDisTorchClipNoDevice


# Backward compatibility alias
override_class_with_distorch = override_class_with_distorch_gguf


# ============================================================================
# STANDARD MULTIGPU WRAPPERS (Device selection without DisTorch)
# ============================================================================

def override_class(cls):
    """Standard MultiGPU device override for UNet/VAE models"""
    from . import set_current_device, get_current_device
    
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
            original_device = get_current_device()
            if device is not None:
                set_current_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            try:
                return out
            finally:
                set_current_device(original_device)

    return NodeOverride

def override_class_offload(cls):
    """Standard MultiGPU device override for UNet/VAE models"""
    from . import set_current_device, set_current_unet_offload_device, get_current_device, get_current_unet_offload_device
    
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            inputs["optional"]["offload_device"] = (devices, {"default": "cpu"})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, offload_device=None, **kwargs):
            original_device = get_current_device()
            original_offload_device = get_current_unet_offload_device()
            if device is not None:
                set_current_device(device)
            if offload_device is not None:
                set_current_unet_offload_device(offload_device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            try:
                return out
            finally:
                set_current_device(original_device)
                set_current_unet_offload_device(original_offload_device)

    return NodeOverride



def override_class_clip(cls):
    """Standard MultiGPU device override for CLIP models (with device kwarg workaround)"""
    from . import set_current_text_encoder_device, get_current_text_encoder_device
    
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
            original_text_device = get_current_text_encoder_device()
            if device is not None:
                set_current_text_encoder_device(device)
            kwargs['device'] = 'default'
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            try:
                return out
            finally:
                set_current_text_encoder_device(original_text_device)

    return NodeOverride


def override_class_clip_no_device(cls):
    """Standard MultiGPU device override for Triple/Quad CLIP models (no device kwarg workaround)"""
    from . import set_current_text_encoder_device, get_current_text_encoder_device
    
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
            original_text_device = get_current_text_encoder_device()
            if device is not None:
                set_current_text_encoder_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            try:
                return out
            finally:
                set_current_text_encoder_device(original_text_device)

    return NodeOverride
