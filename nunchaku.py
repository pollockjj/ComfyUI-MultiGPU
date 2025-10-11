import logging
import comfy.model_management as mm
import folder_paths
from nodes import NODE_CLASS_MAPPINGS
from .device_utils import get_device_list
from .model_management_mgpu import multigpu_memory_log
from comfy.utils import load_torch_file, ProgressBar
import gc
import numpy as np
from accelerate import init_empty_weights
import os
import importlib.util

logger = logging.getLogger("MultiGPU")
class NunchakuQwenImageDiTLoader:
    """ Loader for Nunchaku Qwen-Image models."""
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {"required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The Nunchaku Qwen-Image model."}),
                "compute_device": (devices, {"default": default_device}),
                "offload_device": (devices, {"default": default_device}),
                "num_blocks_on_gpu": ("INT", {"default": 1, "min": 1, "max": 60, "tooltip": ("When CPU offload is enabled, this option determines how many transformer blocks remain on GPU memory. " "Increasing this value decreases CPU RAM usage but increases GPU memory usage.")}),
            },
            "optional": {
                "use_pin_memory": (["enable", "disable"], {"default": "disable", "tooltip": ("Enable this to use pinned memory for transformer blocks when CPU offload is enabled. " "This can improve data transfer speed between CPU and GPU, but may increase system memory usage.")}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "multigpu/Nunchaku"
    TITLE = "Nunchaku Qwen-Image DiT Loader"

    def load_model(self, model_name, compute_device, offload_device, num_blocks_on_gpu, use_pin_memory="disable"):
        from . import set_current_device, set_current_unet_offload_device

        original_module_device = mm.get_torch_device()
        original_module_offload_device = mm.unet_offload_device()

        set_current_device(compute_device)
        set_current_unet_offload_device(offload_device)

        cpu_offload = "enable"

        if offload_device != "cpu":
            use_pin_memory = "disable"
            logger.info("[MultiGPU Nunchaku][NunchakuQwenImageDiTLoaderMultiGPU]Offload device is not CPU, disabling pin memory")

        original_loader = NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"]()
        try:
            return original_loader.load_model(model_name, cpu_offload, num_blocks_on_gpu, use_pin_memory)
        finally:
            set_current_device(original_module_device)
            set_current_unet_offload_device(original_module_offload_device)