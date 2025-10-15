import torch
import logging
import hashlib
import comfy.sd
import comfy.utils
import comfy.model_management as mm
import comfy.model_detection
import comfy.clip_vision
from comfy.sd import VAE, CLIP
from .device_utils import get_device_list, soft_empty_cache_multigpu
from .model_management_mgpu import multigpu_memory_log
from .distorch_2 import register_patched_safetensor_modelpatcher

logger = logging.getLogger("MultiGPU")

checkpoint_device_config = {}
checkpoint_distorch_config = {}

original_load_state_dict_guess_config = None

def patch_load_state_dict_guess_config():
    """Monkey patch comfy.sd.load_state_dict_guess_config with MultiGPU-aware checkpoint loading."""
    global original_load_state_dict_guess_config
    
    if original_load_state_dict_guess_config is not None:
        logger.debug("[MultiGPU Checkpoint] load_state_dict_guess_config is already patched.")
        return
    
    logger.info("[MultiGPU Core Patching] Patching comfy.sd.load_state_dict_guess_config for advanced MultiGPU loading.")
    original_load_state_dict_guess_config = comfy.sd.load_state_dict_guess_config
    comfy.sd.load_state_dict_guess_config = patched_load_state_dict_guess_config

def patched_load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False,
                                        embedding_directory=None, output_model=True, model_options={},
                                        te_model_options={}, metadata=None):
    """Patched checkpoint loader with MultiGPU and DisTorch2 device placement support."""
    from . import set_current_device, set_current_text_encoder_device, get_current_device, get_current_text_encoder_device
    
    sd_size = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))
    config_hash = str(sd_size)
    device_config = checkpoint_device_config.get(config_hash)
    distorch_config = checkpoint_distorch_config.get(config_hash)

    if not device_config and not distorch_config:
        return original_load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options, metadata)

    logger.debug("[MultiGPU Checkpoint] ENTERING Patched Checkpoint Loader")
    logger.debug(f"[MultiGPU Checkpoint] Received Device Config: {device_config}")
    logger.debug(f"[MultiGPU Checkpoint] Received DisTorch2 Config: {distorch_config}")

    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    
    # Capture the current devices at runtime so we can restore them after loading
    original_main_device = get_current_device()
    original_clip_device = get_current_text_encoder_device()

    try:
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
        weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
        model_config = comfy.model_detection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
        
        if model_config is None:
            logger.warning("[MultiGPU] Warning: Not a standard checkpoint file. Trying to load as diffusion model only.")
            # Simplified fallback for non-checkpoints
            set_current_device(device_config.get('unet_device', original_main_device))
            diffusion_model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
            if diffusion_model is None:
                return None
            return (diffusion_model, None, VAE(sd={}), None)

        logger.debug(f"[MultiGPU] Detected Model Config: {type(model_config).__name__}, Parameters: {parameters/10**9:.2f}B")

        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if model_config.scaled_fp8 is not None:
            weight_dtype = None
        
        model_config.custom_operations = model_options.get("custom_operations", None)
        unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))
        if unet_dtype is None:
            unet_dtype = mm.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
        
        unet_compute_device = device_config.get('unet_device', original_main_device)
        manual_cast_dtype = mm.unet_manual_cast(unet_dtype, torch.device(unet_compute_device), model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        logger.info(f"UNet DType: {unet_dtype}, Manual Cast: {manual_cast_dtype}")


        if model_config.clip_vision_prefix is not None and output_clipvision:
            clipvision = comfy.clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

        if output_model:
            unet_compute_device = device_config.get('unet_device', original_main_device)
            set_current_device(unet_compute_device)            
            inital_load_device = mm.unet_inital_load_device(parameters, unet_dtype)

            multigpu_memory_log(f"unet:{config_hash[:8]}", "pre-load")

            model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)

            logger.mgpu_mm_log("Invoking soft_empty_cache_multigpu before UNet ModelPatcher setup")
            soft_empty_cache_multigpu()
            model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=unet_compute_device, offload_device=mm.unet_offload_device())
            multigpu_memory_log(f"unet:{config_hash[:8]}", "post-model")

            if distorch_config and 'unet_allocation' in distorch_config:
                unet_alloc = distorch_config['unet_allocation']
                if unet_alloc:
                    register_patched_safetensor_modelpatcher()
                    inner_model = model_patcher.model
                    inner_model._distorch_v2_meta = {"full_allocation": unet_alloc}
                    logger.info(f"[CHECKPOINT_META] UNET inner_model id=0x{id(inner_model):x}")
                    model._distorch_high_precision_loras = distorch_config.get('high_precision_loras', True)

            model.load_model_weights(sd, diffusion_model_prefix)
            multigpu_memory_log(f"unet:{config_hash[:8]}", "post-weights")

        if output_vae:
            vae_target_device = torch.device(device_config.get('vae_device', original_main_device))
            set_current_device(vae_target_device) # Use main device context for VAE
            multigpu_memory_log(f"vae:{config_hash[:8]}", "pre-load")
            
            vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
            vae_sd = model_config.process_vae_state_dict(vae_sd)
            vae = VAE(sd=vae_sd, metadata=metadata)
            multigpu_memory_log(f"vae:{config_hash[:8]}", "post-load")

        if output_clip:
            clip_target_device = device_config.get('clip_device', original_clip_device)
            set_current_text_encoder_device(clip_target_device)
            
            clip_target = model_config.clip_target(state_dict=sd)
            if clip_target is not None:
                clip_sd = model_config.process_clip_state_dict(sd)
                if len(clip_sd) > 0:
                    logger.debug("[MultiGPU Checkpoint] Invoking soft_empty_cache_multigpu before CLIP construction")
                    multigpu_memory_log(f"clip:{config_hash[:8]}", "pre-load")
                    soft_empty_cache_multigpu()
                    clip_params = comfy.utils.calculate_parameters(clip_sd)
                    clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=clip_params, model_options=te_model_options)

                    if distorch_config and 'clip_allocation' in distorch_config:
                        clip_alloc = distorch_config['clip_allocation']
                        if clip_alloc and hasattr(clip, 'patcher'):
                            register_patched_safetensor_modelpatcher()
                            inner_clip = clip.patcher.model
                            inner_clip._distorch_v2_meta = {"full_allocation": clip_alloc}
                            logger.info(f"[CHECKPOINT_META] CLIP inner_model id=0x{id(inner_clip):x}")
                            clip.patcher.model._distorch_high_precision_loras = distorch_config.get('high_precision_loras', True)

                    m, u = clip.load_sd(clip_sd, full_model=True) # This respects the patched text_encoder_device
                    if len(m) > 0: logger.warning(f"CLIP missing keys: {m}")
                    if len(u) > 0: logger.debug(f"CLIP unexpected keys: {u}")
                    logger.info("CLIP Loaded.")
                    multigpu_memory_log(f"clip:{config_hash[:8]}", "post-load")
                else:
                    logger.warning("No CLIP/text encoder weights in checkpoint.")
            else:
                logger.warning("CLIP target not found in model config.")
        
    finally:
        set_current_device(original_main_device)
        set_current_text_encoder_device(original_clip_device)
        if config_hash in checkpoint_device_config:
            del checkpoint_device_config[config_hash]
        if config_hash in checkpoint_distorch_config:
            del checkpoint_distorch_config[config_hash]
    return (model_patcher, clip, vae, clipvision)

class CheckpointLoaderAdvancedMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]
        
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "unet_device": (devices, {"default": default_device}),
                "clip_device": (devices, {"default": default_device}),
                "vae_device": (devices, {"default": default_device}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "multigpu"
    TITLE = "Checkpoint Loader Advanced (MultiGPU)"
    
    def load_checkpoint(self, ckpt_name, unet_device, clip_device, vae_device):
        patch_load_state_dict_guess_config()
        
        import folder_paths
        import comfy.utils
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        sd = comfy.utils.load_torch_file(ckpt_path)
        sd_size = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))
        config_hash = str(sd_size)
        
        checkpoint_device_config[config_hash] = {
            'unet_device': unet_device, 'clip_device': clip_device, 'vae_device': vae_device
        }
        
        # Load using standard loader, our patch will intercept
        from nodes import CheckpointLoaderSimple
        return CheckpointLoaderSimple().load_checkpoint(ckpt_name)


class CheckpointLoaderAdvancedDisTorch2MultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        devices = get_device_list()
        compute_device = devices[1] if len(devices) > 1 else devices[0]
        
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "unet_compute_device": (devices, {"default": compute_device}),
                "unet_virtual_vram_gb": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "unet_donor_device": ("STRING", {"default": "cpu"}),
                "clip_compute_device": (devices, {"default": "cpu"}),
                "clip_virtual_vram_gb": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "clip_donor_device": ("STRING", {"default": "cpu"}),
                "vae_device": (devices, {"default": compute_device}),
            }, "optional": {
                "unet_expert_mode_allocations": ("STRING", {"multiline": False, "default": ""}),
                "clip_expert_mode_allocations": ("STRING", {"multiline": False, "default": ""}),
                "high_precision_loras": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "multigpu/distorch_2"
    TITLE = "Checkpoint Loader Advanced (DisTorch2)"
    
    def load_checkpoint(self, ckpt_name, unet_compute_device, unet_virtual_vram_gb, unet_donor_device,
                       clip_compute_device, clip_virtual_vram_gb, clip_donor_device, vae_device,
                       unet_expert_mode_allocations="", clip_expert_mode_allocations="", high_precision_loras=True):
        
        patch_load_state_dict_guess_config()        
        
        import folder_paths
        import comfy.utils
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        sd = comfy.utils.load_torch_file(ckpt_path)
        sd_size = sum(p.numel() for p in sd.values() if hasattr(p, 'numel'))
        config_hash = str(sd_size)
        
        checkpoint_device_config[config_hash] = {
            'unet_device': unet_compute_device,
            'clip_device': clip_compute_device,
            'vae_device': vae_device
        }

        unet_vram_str = ""
        if unet_virtual_vram_gb > 0:
            unet_vram_str = f"{unet_compute_device};{unet_virtual_vram_gb};{unet_donor_device}"
        elif unet_expert_mode_allocations:
            unet_vram_str = unet_compute_device
        unet_alloc = f"{unet_expert_mode_allocations}#{unet_vram_str}" if unet_expert_mode_allocations or unet_vram_str else ""
        
        clip_vram_str = ""
        if clip_virtual_vram_gb > 0:
            clip_vram_str = f"{clip_compute_device};{clip_virtual_vram_gb};{clip_donor_device}"
        elif clip_expert_mode_allocations:
            clip_vram_str = clip_compute_device
        clip_alloc = f"{clip_expert_mode_allocations}#{clip_vram_str}" if clip_expert_mode_allocations or clip_vram_str else ""

        checkpoint_distorch_config[config_hash] = {
            'unet_allocation': unet_alloc,
            'clip_allocation': clip_alloc,
            'high_precision_loras': high_precision_loras,
            'unet_settings': hashlib.sha256(f"{unet_alloc}{high_precision_loras}".encode()).hexdigest(),
            'clip_settings': hashlib.sha256(f"{clip_alloc}{high_precision_loras}".encode()).hexdigest(),
        }
        
        from nodes import CheckpointLoaderSimple
        return CheckpointLoaderSimple().load_checkpoint(ckpt_name)
