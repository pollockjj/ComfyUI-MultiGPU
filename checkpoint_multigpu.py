"""
Advanced Checkpoint Loaders for MultiGPU
Provides device-specific and DisTorch2 sharding for checkpoint components
"""

import torch
import logging
import hashlib
import copy
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from .device_utils import get_device_list
from .distorch_2 import safetensor_allocation_store, create_safetensor_model_hash

logger = logging.getLogger("MultiGPU")

# Store checkpoint loading configurations
checkpoint_device_config = {}
checkpoint_distorch_config = {}

# Store the original function
original_load_state_dict_guess_config = None

def create_checkpoint_config_hash(checkpoint_name, config_str):
    """Create a unique hash for checkpoint configuration"""
    identifier = f"{checkpoint_name}_{config_str}"
    return hashlib.sha256(identifier.encode()).hexdigest()

def patch_load_state_dict_guess_config():
    """Monkey patch the load_state_dict_guess_config function to support per-component device selection"""
    global original_load_state_dict_guess_config
    
    if original_load_state_dict_guess_config is not None:
        return  # Already patched
    
    original_load_state_dict_guess_config = comfy.sd.load_state_dict_guess_config
    
    def patched_load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, 
                                            embedding_directory=None, output_model=True, model_options={}, 
                                            te_model_options={}, metadata=None):
        
        # Import here to avoid circular imports
        from . import set_current_device, set_current_text_encoder_device, current_device, current_text_encoder_device
        
        # Check if we have a device configuration for this checkpoint
        # We use the state dict size as a simple identifier
        sd_size = sum(t.numel() for t in sd.values() if hasattr(t, 'numel'))
        config_hash = str(sd_size)
        
        device_config = checkpoint_device_config.get(config_hash)
        distorch_config = checkpoint_distorch_config.get(config_hash)
        
        if device_config or distorch_config:
            logger.info(f"[MultiGPU] Using custom device configuration for checkpoint")
            
            # Save original devices
            original_unet_device = current_device
            original_clip_device = current_text_encoder_device
            
            # Handle UNet device/DisTorch config
            if device_config and 'unet_device' in device_config:
                set_current_device(device_config['unet_device'])
                logger.info(f"[MultiGPU] Setting UNet device to: {device_config['unet_device']}")
            
            # Apply DisTorch2 config for UNet if present
            if distorch_config and 'unet_allocation' in distorch_config:
                # We'll store this for when the model patcher is created
                logger.info(f"[MultiGPU] DisTorch2 UNet allocation will be applied: {distorch_config['unet_allocation']}")
            
            # Call original function to load the checkpoint
            result = original_load_state_dict_guess_config(
                sd, output_vae=output_vae, output_clip=output_clip, output_clipvision=output_clipvision,
                embedding_directory=embedding_directory, output_model=output_model, 
                model_options=model_options, te_model_options=te_model_options, metadata=metadata
            )
            
            model_patcher, clip, vae, clipvision = result
            
            # Apply DisTorch2 configurations after loading
            if distorch_config:
                if model_patcher and 'unet_allocation' in distorch_config:
                    model_hash = create_safetensor_model_hash(model_patcher, "checkpoint_loader")
                    safetensor_allocation_store[model_hash] = distorch_config['unet_allocation']
                    if 'unet_settings' in distorch_config:
                        from .distorch_2 import safetensor_settings_store
                        safetensor_settings_store[model_hash] = distorch_config['unet_settings']
                    logger.info(f"[MultiGPU] Applied DisTorch2 config to UNet: {model_hash[:8]}")
                
                if clip and 'clip_allocation' in distorch_config:
                    # For CLIP, we need to get the model from the CLIP object
                    if hasattr(clip, 'patcher'):
                        clip_hash = create_safetensor_model_hash(clip.patcher, "checkpoint_loader_clip")
                        safetensor_allocation_store[clip_hash] = distorch_config['clip_allocation']
                        if 'clip_settings' in distorch_config:
                            from .distorch_2 import safetensor_settings_store
                            safetensor_settings_store[clip_hash] = distorch_config['clip_settings']
                        logger.info(f"[MultiGPU] Applied DisTorch2 config to CLIP: {clip_hash[:8]}")
            
            # Handle CLIP device
            if device_config and 'clip_device' in device_config and clip:
                set_current_text_encoder_device(device_config['clip_device'])
                logger.info(f"[MultiGPU] Setting CLIP device to: {device_config['clip_device']}")
                # Force CLIP to load on the specified device
                if hasattr(clip, 'patcher'):
                    clip.patcher.load(force_patch_weights=True)
            
            # Handle VAE device
            if device_config and 'vae_device' in device_config and vae:
                vae_device = torch.device(device_config['vae_device'])
                logger.info(f"[MultiGPU] Setting VAE device to: {device_config['vae_device']}")
                # Move VAE to specified device
                if hasattr(vae, 'first_stage_model'):
                    vae.first_stage_model = vae.first_stage_model.to(vae_device)
            
            # Clean up stored configs
            if config_hash in checkpoint_device_config:
                del checkpoint_device_config[config_hash]
            if config_hash in checkpoint_distorch_config:
                del checkpoint_distorch_config[config_hash]
            
            return result
        else:
            # No custom config, use original behavior
            return original_load_state_dict_guess_config(
                sd, output_vae=output_vae, output_clip=output_clip, output_clipvision=output_clipvision,
                embedding_directory=embedding_directory, output_model=output_model, 
                model_options=model_options, te_model_options=te_model_options, metadata=metadata
            )
    
    # Apply the patch
    comfy.sd.load_state_dict_guess_config = patched_load_state_dict_guess_config
    logger.info("[MultiGPU] Successfully patched load_state_dict_guess_config")


class CheckpointLoaderAdvancedMultiGPU:
    """
    Checkpoint loader that allows loading UNet, CLIP, and VAE to different devices
    """
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
        # Apply the patch if not already applied
        patch_load_state_dict_guess_config()
        
        # Store device configuration
        import folder_paths
        import comfy.utils
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        sd = comfy.utils.load_torch_file(ckpt_path)
        
        # Use state dict size as identifier
        sd_size = sum(t.numel() for t in sd.values() if hasattr(t, 'numel'))
        config_hash = str(sd_size)
        
        # Store the device configuration
        checkpoint_device_config[config_hash] = {
            'unet_device': unet_device,
            'clip_device': clip_device,
            'vae_device': vae_device
        }
        
        logger.info(f"[MultiGPU] CheckpointLoaderAdvanced configured - UNet: {unet_device}, CLIP: {clip_device}, VAE: {vae_device}")
        
        # Load the checkpoint - our patched function will handle device placement
        from nodes import CheckpointLoaderSimple
        loader = CheckpointLoaderSimple()
        return loader.load_checkpoint(ckpt_name)


class CheckpointLoaderAdvancedDisTorch2MultiGPU:
    """
    Checkpoint loader with full DisTorch2 sharding for UNet and CLIP, device selection for VAE
    """
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        devices = get_device_list()
        compute_device = devices[1] if len(devices) > 1 else devices[0]
        
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                # UNet DisTorch2 settings
                "unet_compute_device": (devices, {"default": compute_device}),
                "unet_virtual_vram_gb": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "unet_donor_device": (devices, {"default": "cpu"}),
                # CLIP DisTorch2 settings
                "clip_compute_device": (devices, {"default": compute_device}),
                "clip_virtual_vram_gb": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "clip_donor_device": (devices, {"default": "cpu"}),
                # VAE simple device
                "vae_device": (devices, {"default": compute_device}),
            },
            "optional": {
                "unet_expert_mode_allocations": ("STRING", {"multiline": False, "default": ""}),
                "clip_expert_mode_allocations": ("STRING", {"multiline": False, "default": ""}),
                "high_precision_loras": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "multigpu/distorch_2"
    TITLE = "Checkpoint Loader Advanced (DisTorch2)"
    
    def load_checkpoint(self, ckpt_name, 
                       unet_compute_device, unet_virtual_vram_gb, unet_donor_device,
                       clip_compute_device, clip_virtual_vram_gb, clip_donor_device,
                       vae_device,
                       unet_expert_mode_allocations="", clip_expert_mode_allocations="",
                       high_precision_loras=True):
        
        # Apply the patch if not already applied
        patch_load_state_dict_guess_config()
        
        # Register DisTorch2 model patcher
        from .distorch_2 import register_patched_safetensor_modelpatcher
        register_patched_safetensor_modelpatcher()
        
        # Store device configuration
        import folder_paths
        import comfy.utils
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        sd = comfy.utils.load_torch_file(ckpt_path)
        
        # Use state dict size as identifier
        sd_size = sum(t.numel() for t in sd.values() if hasattr(t, 'numel'))
        config_hash = str(sd_size)
        
        # Store device configuration
        checkpoint_device_config[config_hash] = {
            'unet_device': unet_compute_device,
            'clip_device': clip_compute_device,
            'vae_device': vae_device
        }
        
        # Build DisTorch2 allocation strings
        unet_vram_string = ""
        if unet_virtual_vram_gb > 0:
            unet_vram_string = f"{unet_compute_device};{unet_virtual_vram_gb};{unet_donor_device}"
        elif unet_expert_mode_allocations:
            unet_vram_string = unet_compute_device
        
        unet_allocation = f"{unet_expert_mode_allocations}#{unet_vram_string}" if unet_expert_mode_allocations or unet_vram_string else ""
        
        clip_vram_string = ""
        if clip_virtual_vram_gb > 0:
            clip_vram_string = f"{clip_compute_device};{clip_virtual_vram_gb};{clip_donor_device}"
        elif clip_expert_mode_allocations:
            clip_vram_string = clip_compute_device
        
        clip_allocation = f"{clip_expert_mode_allocations}#{clip_vram_string}" if clip_expert_mode_allocations or clip_vram_string else ""
        
        # Create settings hashes for DisTorch2
        unet_settings_str = f"{unet_compute_device}{unet_virtual_vram_gb}{unet_donor_device}{unet_expert_mode_allocations}{high_precision_loras}"
        unet_settings_hash = hashlib.sha256(unet_settings_str.encode()).hexdigest()
        
        clip_settings_str = f"{clip_compute_device}{clip_virtual_vram_gb}{clip_donor_device}{clip_expert_mode_allocations}{high_precision_loras}"
        clip_settings_hash = hashlib.sha256(clip_settings_str.encode()).hexdigest()
        
        # Store DisTorch2 configuration
        checkpoint_distorch_config[config_hash] = {
            'unet_allocation': unet_allocation,
            'unet_settings': unet_settings_hash,
            'clip_allocation': clip_allocation,
            'clip_settings': clip_settings_hash,
            'high_precision_loras': high_precision_loras
        }
        
        logger.info(f"[MultiGPU] CheckpointLoaderDisTorch2 configured:")
        logger.info(f"  UNet: compute={unet_compute_device}, vram={unet_virtual_vram_gb}GB, donor={unet_donor_device}")
        logger.info(f"  CLIP: compute={clip_compute_device}, vram={clip_virtual_vram_gb}GB, donor={clip_donor_device}")
        logger.info(f"  VAE: device={vae_device}")
        
        # Load the checkpoint - our patched function will handle device placement and DisTorch2
        from nodes import CheckpointLoaderSimple
        loader = CheckpointLoaderSimple()
        
        # Set high precision loras flag
        result = loader.load_checkpoint(ckpt_name)
        
        # Store high_precision_loras in the models
        model_patcher, clip, vae = result
        if model_patcher and hasattr(model_patcher, 'model'):
            model_patcher.model._distorch_high_precision_loras = high_precision_loras
        if clip and hasattr(clip, 'patcher') and hasattr(clip.patcher, 'model'):
            clip.patcher.model._distorch_high_precision_loras = high_precision_loras
        
        return result
