import logging
import torch
import sys
import inspect
import copy
import folder_paths
import comfy.model_management as mm
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


class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
            "quantization": (["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e4m3fn_scaled", "fp8_e4m3fn_scaled_fast", "fp8_e5m2", "fp8_e5m2_fast", "fp8_e5m2_scaled", "fp8_e5m2_scaled_fast"], {"default": "disabled", 
                            "tooltip": "Optional quantization method, 'disabled' acts as autoselect based by weights. Scaled modes only work with matching weights, _fast modes (fp8 matmul) require CUDA compute capability >= 8.9 (NVIDIA 4000 series and up), e4m3fn generally can not be torch.compiled on compute capability < 8.9 (3000 series and under)"}),
            "load_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            "compute_device": (devices, {"default": default_device}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "sageattn_3",
                    "radial_sage_attention",
                    ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None, "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
                "extra_model": ("VACEPATH", {"default": None, "tooltip": "Extra model to add to the main model, ie. VACE or MTV Crafter"}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None, "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None, "tooltip": "Multitalk model"}),
                "fantasyportrait_model": ("FANTASYPORTRAITMODEL", {"default": None, "tooltip": "FantasyPortrait model"}),
                "rms_norm_function": (["default", "pytorch"], {"default": "default", "tooltip": "RMSNorm function to use, 'pytorch' is the new native torch RMSNorm, which is faster (when not using torch.compile mostly) but changes results slightly. 'default' is the original WanRMSNorm"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL", "MULTIGPUDEVICE",)
    RETURN_NAMES = ("model", "compute_device",)
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"

    def loadmodel(self, model, base_precision, compute_device, quantization, load_device,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, 
                  vram_management_args=None, extra_model=None, vace_model=None,
                  fantasytalking_model=None, multitalk_model=None, fantasyportrait_model=None, 
                  rms_norm_function="default"):
        from . import set_current_device

        set_current_device(compute_device)      
        compute_device_to_be_patched = mm.get_torch_device()

        original_loader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        loader_module = inspect.getmodule(original_loader)

        original_module_device = loader_module.device

        loader_module.device = compute_device_to_be_patched

        result = original_loader.loadmodel(model, base_precision, load_device, quantization, compile_args, attention_mode, block_swap_args, lora, vram_management_args, extra_model=extra_model, 
            vace_model=vace_model, fantasytalking_model=fantasytalking_model, multitalk_model=multitalk_model, fantasyportrait_model=fantasyportrait_model, rms_norm_function=rms_norm_function,)

        loader_module.device = original_module_device

        patcher = result[0]

        return (patcher, compute_device)

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
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. For rudimentary prompt travel you can input multiple prompts separated by '|', they will be equally spread over the video length"

    def process(self, positive_prompt, negative_prompt, t5=None, load_device=None,force_offload=True, model_to_offload=None, use_disk_cache=False):
        from . import set_current_device

        set_current_device(load_device)

        if load_device == "cpu":
            device = "cpu"
        else:
            device = "gpu"

        if t5 is not None:
            text_encoder = t5[0]
        else:
            text_encoder = None

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncodeMulitiGPU] current_device set to: {load_device}")
        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncodeMulitiGPU] device set to: {device}")

        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        prompt_embeds_dict = original_encoder.process(positive_prompt, negative_prompt, text_encoder, force_offload, model_to_offload, use_disk_cache, device)
        return (prompt_embeds_dict)

    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        original_parser = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        return original_parser.parse_prompt_weights(prompt)

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

        return text_encoder, device


class WanVideoTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
                "precision": (["fp32", "bf16"], {"default": "bf16"}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
                "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
                "use_disk_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
                "load_device": (devices, {"default": default_device}
                ),
            },
            "optional": {
                "extender_args": ("WANVIDEOPROMPTEXTENDER_ARGS", {"tooltip": "Use this node to extend the prompt with additional text."}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds", "positive_prompt")
    OUTPUT_TOOLTIPS = ("The text embeddings for both prompts", "The text embeddings for the negative prompt only (for NAG)", "Positive prompt to display prompt extender results")
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = """Encodes text prompts into text embeddings. This node loads and completely unloads the T5 after done, leaving no VRAM or RAM imprint."""


    def process(self, model_name, precision, positive_prompt, negative_prompt, quantization='disabled', use_disk_cache=True, load_device=None, extender_args=None):
        from . import set_current_device

        if load_device is not None:
            set_current_device(load_device)

        if load_device == "cpu":
            device = "cpu"
        else:
            device = "gpu"

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncodeCachedMulitiGPU] current_device set to: {load_device}")
        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncodeCachedMulitiGPU] device set to: {device}")

        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
        prompt_embeds_dict, negative_text_embeds, positive_prompt_out = original_encoder.process(model_name, precision, positive_prompt, negative_prompt, quantization, use_disk_cache, device, extender_args)

        return prompt_embeds_dict, negative_text_embeds, positive_prompt_out

class WanVideoTextEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "load_device": ("MULTIGPUDEVICE",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use, under the custom_nodes/ComfyUI-WanVideoWrapper/text_embed_cache directory"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Encodes text prompt into text embedding."

    def process(self, prompt, t5=None, load_device=None, force_offload=True, model_to_offload=None, use_disk_cache=False):
        from . import set_current_device

        if load_device is not None:
            set_current_device(load_device)

        if load_device == "cpu":
            device = "cpu"
        else:
            device = "gpu"

        if t5 is not None:
            text_encoder = t5[0]
        else:
            text_encoder = None

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncodeSingleMulitiGPU] current_device set to: {load_device}")
        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTextEncodeSingleMulitiGPU] device set to: {device}")

        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncodeSingle"]()
        prompt_embeds_dict = original_encoder.process(prompt, text_encoder, force_offload, model_to_offload, use_disk_cache, device)
        return (prompt_embeds_dict)

class WanVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "load_device": (devices, {"default": default_device}),
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
                "compile_args": ("WANCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("WANVAE", "MULTIGPUDEVICE",)
    RETURN_NAMES = ("vae", "load_device",)
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, load_device=None, precision="fp16", compile_args=None):
        from . import set_current_device

        if load_device is not None:
            set_current_device(load_device)

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoVAELoaderMultiGPU] load_device set to: {load_device}")

        original_loader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        vae_model = original_loader.loadmodel(model_name, precision, compile_args)

        # Return both the VAE model AND the selected device for device propagation
        return vae_model, load_device

class WanVideoTinyVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae_approx"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae_approx'"}),
            },
            "optional": {
                "load_device": (devices, {"default": default_device}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}), 
                "parallel": ("BOOLEAN", {"default": False, "tooltip": "uses more memory but is faster"}),
            }
        }

    RETURN_TYPES = ("WANVAE","MULTIGPUDEVICE")
    RETURN_NAMES = ("vae", "load_device")
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae_approx'"

    def loadmodel(self, model_name, load_device=None, precision="fp16", parallel=False):
        from . import set_current_device

        if load_device is not None:
            set_current_device(load_device)

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoTinyVAELoader] load_device set to: {load_device}")

        original_loader = NODE_CLASS_MAPPINGS["WanVideoTinyVAELoader"]()
        vae_model = original_loader.loadmodel(model_name, precision, parallel)

        # Return both the VAE model AND the selected device for device propagation
        return vae_model, load_device


class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        base_inputs = copy.deepcopy(NODE_CLASS_MAPPINGS["WanVideoBlockSwap"].INPUT_TYPES())
        devices = get_device_list()
        default_device = "cpu" if "cpu" in devices else devices[0]
        base_inputs.setdefault("optional", {})
        base_inputs["optional"]["swap_device"] = (
            devices,
            {
                "default": default_device,
                "tooltip": "Device that receives swapped transformer blocks",
            },
        )
        return base_inputs

    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Extends Wan block swap with explicit device selection"

    def setargs(self, swap_device=None, **kwargs):
        block_swap_config = dict(kwargs)
        if swap_device is not None:
            block_swap_config["swap_device"] = str(swap_device)
        return (block_swap_config,)

class WanVideoImageToVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "load_device": ("MULTIGPUDEVICE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "end_image": ("IMAGE", {"tooltip": "end frame"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"

    def process(self, width, height, num_frames, force_offload, noise_aug_strength, 
                    start_latent_strength, end_latent_strength, start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False, 
                    temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None, vae=None, load_device=None):
        from . import set_current_device

        set_current_device(load_device)
        
        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoImageToVideoEncodeMultiGPU] load device: {load_device}")

        device = mm.get_torch_device()
        PATCH_SIZE = (1, 2, 2)
        offload_device = mm.unet_offload_device()

        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoImageToVideoEncodeMultiGPU] torch device: {device}")

        if vae is not None:
            vae = vae[0]
        
        if start_image is None and end_image is None and add_cond_latents is None:
            return WanVideoEmptyEmbeds().process(
                num_frames, width, height, control_embeds=control_embeds, extra_latents=extra_latents,
            )
        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        H = height
        W = width
        
        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        if start_image is None and end_image is not None:
            fun_or_fl2v_model = True # end image alone only works with this option

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1  # First frame
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, vae.dtype)

        # Repeat first frame and optionally end frame
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
        if end_image is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1) # T, C, H, W
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
        mask = mask.movedim(1, 2)[0]# C, T, H, W

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            start_image = start_image[..., :3]
            if start_image.shape[1] != H or start_image.shape[2] != W:
                resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_image = start_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_image = resized_start_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)
        
        if end_image is not None:
            end_image = end_image[..., :3]
            if end_image.shape[1] != H or end_image.shape[2] != W:
                resized_end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_end_image = end_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_end_image = resized_end_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)
            
        # Concatenate image with zero frames and encode
        if temporal_mask is None:
            if start_image is not None and end_image is None:
                zero_frames = torch.zeros(3, num_frames-start_image.shape[0], H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames], dim=1)
                del resized_start_image, zero_frames
            elif start_image is None and end_image is not None:
                zero_frames = torch.zeros(3, num_frames-end_image.shape[0], H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
                del zero_frames
            elif start_image is None and end_image is None:
                concatenated = torch.zeros(3, num_frames, H, W, device=device, dtype=vae.dtype)
            else:
                if fun_or_fl2v_model:
                    zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, device=device, dtype=vae.dtype)
                else:
                    zero_frames = torch.zeros(3, num_frames-1, H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
                del resized_start_image, zero_frames
        else:
            temporal_mask = common_upscale(temporal_mask.unsqueeze(1), W, H, "nearest", "disabled").squeeze(1)
            concatenated = resized_start_image[:,:num_frames].to(vae.dtype) * temporal_mask[:num_frames].unsqueeze(0).to(vae.dtype)
            del resized_start_image, temporal_mask

        mm.soft_empty_cache()
        gc.collect()

        vae.to(device)
        y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model),tiled=tiled_vae)[0]
        del concatenated

        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        y[:, :1] *= start_latent_strength
        y[:, -1:] *= end_latent_strength

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)
        
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask
        }

        return (image_embeds,)

class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "load_device": ("MULTIGPUDEVICE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                    "optional": {
                        "normalization": (["default", "minmax"], {"advanced": True}),
                    }
                }

    @classmethod
    def VALIDATE_INPUTS(s, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if tile_x <= tile_stride_x:
            return "Tile width must be larger than the tile stride width."
        if tile_y <= tile_stride_y:
            return "Tile height must be larger than the tile stride height."
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "multigpu/WanVideoWrapper"

    def decode(self, vae, load_device, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization="default"):
        from . import set_current_device

        set_current_device(load_device)      
        compute_device_to_be_patched = mm.get_torch_device()
        
        logger.info(f"[MultiGPU WanVideoWrapper][WanVideoDecodeMultiGPU] load device: {load_device}")

        original_loader = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        loader_module = inspect.getmodule(original_loader)

        original_module_device = loader_module.device

        loader_module.device = compute_device_to_be_patched

        result = original_loader.decode(vae[0], samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization)

        loader_module.device = original_module_device

        decode = result[0]

        return (decode,)


class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        # Get original inputs and add our device input
        original_types = NODE_CLASS_MAPPINGS["WanVideoSampler"].INPUT_TYPES()
        original_types["required"]["compute_device"] = ("MULTIGPUDEVICE",)
        return original_types
    
    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware sampler that ensures correct device for each model"
    
    def process(self, model, compute_device, **kwargs):
        from . import set_current_device
        logger.info(
            f"[MultiGPU WanVideoSampler] Received request to process on: {compute_device}"
        )

        patcher = model
        transformer = None
        if hasattr(patcher, "model"):
            transformer = getattr(patcher.model, "diffusion_model", None)

        if compute_device:
            target_device = torch.device(compute_device)
            set_current_device(target_device)
        else:
            target_device = mm.get_torch_device()

        normalized_swap_device = None
        transformer_options = {}
        if hasattr(patcher, "model_options"):
            transformer_options = patcher.model_options.get("transformer_options", {})
        block_swap_args = transformer_options.get("block_swap_args") if transformer_options else None
        if block_swap_args:
            swap_label = block_swap_args.get("resolved_swap_device") or block_swap_args.get("swap_device")
            if swap_label:
                try:
                    normalized_swap_device = torch.device(str(swap_label))
                except (TypeError, ValueError):
                    logger.warning(
                        "[MultiGPU WanVideoSampler] Invalid swap device '%s', leaving sampler offload unchanged",
                        swap_label,
                    )
                    normalized_swap_device = None

        original_sampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        sampler_module = inspect.getmodule(original_sampler)

        original_module_device = None
        original_module_offload = None
        had_sampler_offload_attr = False
        if sampler_module is not None:
            original_module_device = getattr(sampler_module, "device", None)
            had_sampler_offload_attr = hasattr(sampler_module, "offload_device")
            original_module_offload = getattr(sampler_module, "offload_device", None)
            setattr(sampler_module, "device", target_device)
            if normalized_swap_device is not None:
                setattr(sampler_module, "offload_device", normalized_swap_device)
            elif compute_device == "cpu":
                setattr(sampler_module, "offload_device", target_device)
        else:
            logger.error("[MultiGPU WanVideoSampler] Unable to resolve sampler module for device patching.")

        if transformer is not None and normalized_swap_device is not None:
            transformer.offload_device = normalized_swap_device
            transformer.cache_device = normalized_swap_device

        try:
            return original_sampler.process(model=patcher, **kwargs)
        finally:
            if sampler_module is not None:
                if original_module_device is not None:
                    setattr(sampler_module, "device", original_module_device)
                if had_sampler_offload_attr:
                    setattr(sampler_module, "offload_device", original_module_offload)
                else:
                    try:
                        delattr(sampler_module, "offload_device")
                    except AttributeError:
                        pass
