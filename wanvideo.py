import logging
import torch
import sys
import inspect
import folder_paths
import comfy.model_management as mm
from nodes import NODE_CLASS_MAPPINGS
from .device_utils import get_device_list
from .model_management_mgpu import multigpu_memory_log

logger = logging.getLogger("MultiGPU")


class LoadWanVideoT5TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
                "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "device": (devices, {"default": default_device}),
                "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            }
        }

    RETURN_TYPES = ("WANTEXTENCODER", "STRING")
    RETURN_NAMES = ("wan_t5_model", "device")
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Loads Wan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, model_name, precision, device=None, quantization="disabled"):
        from . import set_current_device

        if device is not None:
            set_current_device(device)
        
        if device == "cpu":
            load_device = "offload_device"
        else:
            load_device = "main_device"

        logger.info(f"[MultiGPU WanVideoWrapper] current_device set to: {device}")
        logger.info(f"[MultiGPU WanVideoWrapper] load_device set to: {load_device}")

        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        text_encoder = original_loader.loadmodel(model_name, precision, load_device, quantization)

        # Return both the text encoder AND the selected device
        return text_encoder, device
