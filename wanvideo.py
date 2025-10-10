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


scheduler_list = [
    "unipc", "unipc/beta",
    "dpm++", "dpm++/beta",
    "dpm++_sde", "dpm++_sde/beta",
    "euler", "euler/beta",
    "deis",
    "lcm", "lcm/beta",
    "res_multistep",
    "flowmatch_causvid",
    "flowmatch_distill",
    "flowmatch_pusa",
    "multitalk",
    "sa_ode_stable"
]

rope_functions = ["default", "comfy", "comfy_chunked"]

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

    def loadmodel(self, model, base_precision, compute_device, quantization, load_device, **kwargs):
        from . import set_current_device

        original_loader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        loader_module = inspect.getmodule(original_loader)
        original_module_device = loader_module.device

        set_current_device(compute_device)      
        compute_device_to_be_patched = mm.get_torch_device()

        loader_module.device = compute_device_to_be_patched

        result = original_loader.loadmodel(model, base_precision, load_device, quantization, **kwargs,)

        patcher = result[0]

        try:
            return (patcher, compute_device)

        finally:
            loader_module.device = original_module_device

class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "compute_device": ("MULTIGPUDEVICE",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (scheduler_list, {"default": "unipc",}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", ),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS", ),
                "rope_function": (rope_functions, {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "sigmas": ("SIGMAS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
                "freeinit_args": ("FREEINITARGS", ),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Start step for the sampling, 0 means full sampling, otherwise samples only from this step"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1, "tooltip": "End step for the sampling, -1 means full sampling, otherwise samples only until this step"}),
                "add_noise_to_samples": ("BOOLEAN", {"default": False, "tooltip": "Add noise to the samples before sampling, needed for video2video sampling when starting from clean video"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware sampler that ensures correct device for each model"
    
    def process(self, model, compute_device, **kwargs):
        from . import set_current_device
        
        original_sampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        sampler_module = inspect.getmodule(original_sampler)

        original_module_device = sampler_module.device
        original_module_offload_device = sampler_module.offload_device

        set_current_device(compute_device)
        compute_device_to_be_patched = mm.get_torch_device()
        sampler_module.device = compute_device_to_be_patched

        transformer = model.model.diffusion_model
        transformer_options = model.model_options.get("transformer_options", {})
        block_swap_args = transformer_options.get("block_swap_args")

        multi_gpu_block_swap = block_swap_args is not None and "swap_device" in block_swap_args
        offload_device_to_be_patched = None
        if multi_gpu_block_swap:
            swap_label = block_swap_args.get("swap_device")
            logger.info(f"[MultiGPU WanVideoWrapper][WanVideoSamplerMultiGPU] block swap enabled, swap device: {swap_label}")
            offload_device_to_be_patched = torch.device(str(swap_label))
            sampler_module.offload_device = offload_device_to_be_patched

        if transformer is not None and offload_device_to_be_patched is not None:
            transformer.offload_device = offload_device_to_be_patched
            transformer.cache_device = offload_device_to_be_patched

        try:
            return original_sampler.process(model, **kwargs)
        finally:
            sampler_module.device = original_module_device
            sampler_module.offload_device = original_module_offload_device

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

        text_encoder = t5[0]

        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        prompt_embeds_dict = original_encoder.process(positive_prompt, negative_prompt, text_encoder, force_offload, model_to_offload, use_disk_cache, device)
        return (prompt_embeds_dict)

    def parse_prompt_weights(self, prompt):
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

        set_current_device(device)
        
        if device == "cpu":
            load_device = "offload_device"
        else:
            load_device = "main_device"

        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        text_encoder = original_loader.loadmodel(model_name, precision, load_device, quantization)

        return text_encoder, device

class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("clip_vision") + folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/clip_vision'"}),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            },
            "optional": {
                "device": (devices, {"default": default_device}),
            }
        }

    RETURN_TYPES = ("CLIP_VISION", "MULTIGPUDEVICE")
    RETURN_NAMES = ("wan_clip_vision", "load_device")
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "Loads Wan clip_vision model from 'ComfyUI/models/clip_vision'"

    def loadmodel(self, model_name, precision, device=None):
        from . import set_current_device

        set_current_device(device)
        
        if device == "cpu":
            load_device = "offload_device"
        else:
            load_device = "main_device"

        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoClipTextEncoder"]()
        clip_model = original_loader.loadmodel(model_name, precision, load_device)

        return clip_model, device

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

        set_current_device(load_device)

        if load_device == "cpu":
            device = "cpu"
        else:
            device = "gpu"

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

        set_current_device(load_device)

        if load_device == "cpu":
            device = "cpu"
        else:
            device = "gpu"

        text_encoder = t5[0]

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

        set_current_device(load_device)

        original_loader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        vae_model = original_loader.loadmodel(model_name, precision, compile_args)

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

        set_current_device(load_device)

        original_loader = NODE_CLASS_MAPPINGS["WanVideoTinyVAELoader"]()
        vae_model = original_loader.loadmodel(model_name, precision, parallel)

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

        original_encoder = NODE_CLASS_MAPPINGS["WanVideoImageToVideoEncode"]()
        encoder_module = inspect.getmodule(original_encoder)

        original_module_device = encoder_module.device
        original_module_offload = encoder_module.offload_device

        set_current_device(load_device)

        compute_device_to_be_patched = mm.get_torch_device()
        encoder_module.device = compute_device_to_be_patched

        encoder_module.offload_device = mm.unet_offload_device()

        inner_vae = vae[0]

        try:
            return original_encoder.process(width, height, num_frames, force_offload, noise_aug_strength, start_latent_strength, end_latent_strength, start_image,
                end_image, control_embeds, fun_or_fl2v_model, temporal_mask, extra_latents, clip_embeds, tiled_vae, add_cond_latents, inner_vae,)
        finally:
            encoder_module.device = original_module_device
            encoder_module.offload_device = original_module_offload

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

        original_decode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        decode_module = inspect.getmodule(original_decode)
        original_module_device = decode_module.device

        set_current_device(load_device)      
        compute_device_to_be_patched = mm.get_torch_device()
        decode_module.device = compute_device_to_be_patched

        try:
            return original_decode.decode(vae[0], samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization)
        finally:
            decode_module.device = original_module_device


class WanVideoVACEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "load_device": ("MULTIGPUDEVICE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"

    def process(self, vae, load_device, width, height, num_frames, strength, vace_start_percent, vace_end_percent, input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, tiled_vae=False):
        from . import set_current_device

        original_encode = NODE_CLASS_MAPPINGS["WanVideoVACEEncode"]()
        encode_module = inspect.getmodule(original_encode)
        original_module_device = encode_module.device

        set_current_device(load_device)
        compute_device_to_be_patched = mm.get_torch_device()
        encode_module.device = compute_device_to_be_patched

        try:
            return original_encode.process(vae[0], width, height, num_frames, strength, vace_start_percent, vace_end_percent, input_frames, ref_images, input_masks, prev_vace_embeds, tiled_vae)
        finally:
            encode_module.device = original_module_device


class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "load_device": ("MULTIGPUDEVICE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride height in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                        "mask": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "multigpu/WanVideoWrapper"

    def encode(self, vae, load_device, image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0, mask=None):
        from . import set_current_device

        original_encode = NODE_CLASS_MAPPINGS["WanVideoEncode"]()
        encode_module = inspect.getmodule(original_encode)
        original_module_device = encode_module.device

        set_current_device(load_device)
        compute_device_to_be_patched = mm.get_torch_device()
        encode_module.device = compute_device_to_be_patched

        try:
            return original_encode.encode(vae[0], image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength, latent_strength, mask)
        finally:
            encode_module.device = original_module_device

class WanVideoClipVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "load_device": ("MULTIGPUDEVICE",),
            "image_1": ("IMAGE", {"tooltip": "Image to encode"}),
            "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}), 
            "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}),
            "crop": (["center", "disabled"], {"default": "center", "tooltip": "Crop image to 224x224 before encoding"}),
            "combine_embeds": (["average", "sum", "concat", "batch"], {"default": "average", "tooltip": "Method to combine multiple clip embeds"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "negative_image": ("IMAGE", {"tooltip": "image to use for uncond"}),
                "tiles": ("INT", {"default": 0, "min": 0, "max": 16, "step": 2, "tooltip": "Use matteo's tiled image encoding for improved accuracy"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ratio of the tile average"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "multigpu/WanVideoWrapper"

    def process(self, clip_vision, load_device, image_1, strength_1, strength_2, force_offload, crop, combine_embeds, image_2=None, negative_image=None, tiles=0, ratio=1.0):
        from . import set_current_device

        original_encode = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
        encode_module = inspect.getmodule(original_encode)
        original_module_device = encode_module.device

        set_current_device(load_device)
        compute_device_to_be_patched = mm.get_torch_device()
        encode_module.device = compute_device_to_be_patched

        try:
            return original_encode.process(clip_vision[0], image_1, strength_1, strength_2, force_offload, crop, combine_embeds, image_2, negative_image, tiles, ratio)
        finally:
            encode_module.device = original_module_device

class WanVideoControlnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model": (folder_paths.get_filename_list("controlnet"), {"tooltip": "These models are loaded from the 'ComfyUI/models/controlnet' -folder",}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
                "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'fp8_e4m3fn_fast_no_ffn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
                "device": (devices, {"default": default_device}),
            },
        }

    RETURN_TYPES = ("WANVIDEOCONTROLNET",)
    RETURN_NAMES = ("controlnet", )
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware ControlNet loader for WanVideo models"

    def loadmodel(self, model, base_precision, load_device, quantization, device):
        from . import set_current_device

        set_current_device(device)
        
        original_loader = NODE_CLASS_MAPPINGS["WanVideoControlnetLoader"]()
        return original_loader.loadmodel(model, base_precision, load_device, quantization)

class FantasyTalkingModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "device": (devices, {"default": default_device}),
            },
        }

    RETURN_TYPES = ("FANTASYTALKINGMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware FantasyTalking model loader"

    def loadmodel(self, model, base_precision, device):
        from . import set_current_device

        set_current_device(device)
        
        original_loader = NODE_CLASS_MAPPINGS["FantasyTalkingModelLoader"]()
        return original_loader.loadmodel(model, base_precision)

class Wav2VecModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model": (folder_paths.get_filename_list("wav2vec2"), {"tooltip": "These models are loaded from the 'ComfyUI/models/wav2vec2' -folder",}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
                "device": (devices, {"default": default_device}),
            },
        }

    RETURN_TYPES = ("WAV2VECMODEL",)
    RETURN_NAMES = ("wav2vec_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware Wav2Vec model loader"

    def loadmodel(self, model, base_precision, load_device, device):
        from . import set_current_device

        set_current_device(device)
        
        original_loader = NODE_CLASS_MAPPINGS["Wav2VecModelLoader"]()
        return original_loader.loadmodel(model, base_precision, load_device)

class DownloadAndLoadWav2VecModel:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model": (
                    [
                    "TencentGameMate/chinese-wav2vec2-base",
                    "facebook/wav2vec2-base-960h"
                    ],
                ),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
                "device": (devices, {"default": default_device}),
            },
        }

    RETURN_TYPES = ("WAV2VECMODEL",)
    RETURN_NAMES = ("wav2vec_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware downloadable Wav2Vec model loader"

    def loadmodel(self, model, base_precision, load_device, device):
        from . import set_current_device

        set_current_device(device)
        
        original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadWav2VecModel"]()
        return original_loader.loadmodel(model, base_precision, load_device)

class WanVideoUni3C_ControlnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        return {
            "required": {
                "model": (folder_paths.get_filename_list("controlnet"), {"tooltip": "These models are loaded from the 'ComfyUI/models/controlnet' -folder",}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
                "device": (devices, {"default": default_device}),
                "attention_mode": ([
                        "sdpa",
                        "sageattn",
                        ], {"default": "sdpa"}),
            },
            "optional": {
                "compile_args": ("WANCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("WANVIDEOCONTROLNET",)
    RETURN_NAMES = ("controlnet", )
    FUNCTION = "loadmodel"
    CATEGORY = "multigpu/WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device, device, quantization, attention_mode, compile_args=None):
        from . import set_current_device

        set_current_device(device)
        
        original_loader = NODE_CLASS_MAPPINGS["WanVideoUni3C_ControlnetLoader"]()
        return original_loader.loadmodel(model, base_precision, load_device, quantization, attention_mode, compile_args)