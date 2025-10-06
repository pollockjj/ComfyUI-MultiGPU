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

    RETURN_TYPES = ("WANTEXTENCODER", "MULTIGPUDEVICE")
    RETURN_NAMES = ("wan_t5_model", "load_device")
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

        logger.info(f"[MultiGPU WanVideoWrapper][LoadWanVideoT5TextEncoder] current_device set to: {device}")
        logger.info(f"[MultiGPU WanVideoWrapper][LoadWanVideoT5TextEncoder] load_device set to: {load_device}")

        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        text_encoder = original_loader.loadmodel(model_name, precision, load_device, quantization)

        # Return both the text encoder AND the selected device
        return text_encoder, device


class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "load_device": ("MULTIGPUDEVICE",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                #"device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Device to run the text encoding on."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. For rudimentary prompt travel you can input multiple prompts separated by '|', they will be equally spread over the video length"


    def process(self, positive_prompt, negative_prompt, t5=None, load_device=None,force_offload=True, model_to_offload=None, use_disk_cache=False):
        from . import set_current_device

        if load_device is not None:
            set_current_device(load_device)

        if load_device == "cpu":
            device = "cpu"
        else:

        if t5 is not None:
            text_encoder = t5[0]
        else:
            text_encoder = None

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncode] current_device set to: {load_device}")
        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncode] device set to: {device}")

        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        prompt_embeds_dict = original_encoder.process(positive_prompt, negative_prompt, text_encoder, force_offload, model_to_offload, use_disk_cache, device)
        return (prompt_embeds_dict)

    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        original_parser = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        return original_parser.parse_prompt_weights(prompt)
