import logging
import torch
import sys
import inspect
import folder_paths
import comfy.model_management as mm

class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        from . import get_device_list
        devices = get_device_list()
        
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"),
                          {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' folder",}),
                "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
                "quantization": (
                    ["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp8_e4m3fn_fast_no_ffn", "fp8_e4m3fn_scaled", "fp8_e5m2_scaled"],
                    {"default": "disabled", "tooltip": "optional quantization method"}
                ),
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0], "tooltip": "Device to load the model to"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "sageattn_3",
                    "flex_attention",
                    "radial_sage_attention",
                ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None, "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
                "vace_model": ("VACEPATH", {"default": None, "tooltip": "VACE model to use when not using model that has it included"}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None, "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None, "tooltip": "Multitalk model"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, device, quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None, vace_model=None, fantasytalking_model=None, multitalk_model=None):
        logging.debug(f"[MultiGPU] WanVideoModelLoader: User selected device: {device}")
        
        selected_device = torch.device(device)
        
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.debug(f"[MultiGPU] Patching WanVideo modules to use {selected_device}")
            
            original_device = getattr(loader_module, 'device', None)
            original_offload = getattr(loader_module, 'offload_device', None)
            
            model_offload_override = getattr(loader_module, '_model_offload_device_override', None)
            
            setattr(loader_module, 'device', selected_device)
            if model_offload_override:
                setattr(loader_module, 'offload_device', model_offload_override)
                logging.debug(f"[MultiGPU] Using model offload override: {model_offload_override}")
            elif device == "cpu":
                setattr(loader_module, 'offload_device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                setattr(nodes_module, 'device', selected_device)
                
                nodes_model_offload_override = getattr(nodes_module, '_model_offload_device_override', None)
                if nodes_model_offload_override:
                    setattr(nodes_module, 'offload_device', nodes_model_offload_override)
                elif device == "cpu":
                    setattr(nodes_module, 'offload_device', selected_device)
                logging.debug(f"[MultiGPU] Both WanVideo modules patched successfully")
            
            logging.debug(f"[MultiGPU] Calling original WanVideo loader")
            result = original_loader.loadmodel(model, base_precision, load_device, quantization,
                                              compile_args, attention_mode, block_swap_args, lora, vram_management_args, vace_model, fantasytalking_model, multitalk_model)
            
            if result and len(result) > 0 and hasattr(result[0], 'model'):
                model_obj = result[0]
                if hasattr(model_obj.model, 'diffusion_model'):
                    transformer = model_obj.model.diffusion_model
                    
                    block_swap_override = getattr(loader_module, '_block_swap_device_override', None)
                    if block_swap_override:
                        transformer.offload_device = block_swap_override
                        logging.debug(f"[MultiGPU] Patched WanVideo transformer for block swap to use: {block_swap_override}")
                    
            logging.info(f"[MultiGPU] WanVideo model loaded on {selected_device}")
            
            return result
        else:
            logging.error(f"[MultiGPU] Could not patch WanVideo modules, falling back")
            return original_loader.loadmodel(model, base_precision, load_device, quantization,
                                            compile_args, attention_mode, block_swap_args, lora, vram_management_args, vace_model, fantasytalking_model, multitalk_model)


class WanVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        from . import get_device_list
        devices = get_device_list()
        
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"),
                               {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0],
                                    "tooltip": "Device to load the VAE to"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"], {"default": "bf16"}),
                "compile_args": ("WANCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model with explicit device selection"

    def loadmodel(self, model_name, device, precision="bf16", compile_args=None):
        logging.debug(f"[MultiGPU] WanVideoVAELoader: User selected device: {device}")
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            selected_device = torch.device(device)
            logging.debug(f"[MultiGPU] Patching WanVideo VAE modules to use {selected_device}")
            
            setattr(loader_module, 'offload_device', selected_device)
            setattr(loader_module, 'device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                setattr(nodes_module, 'device', selected_device)
                setattr(nodes_module, 'offload_device', selected_device)
            
            result = original_loader.loadmodel(model_name, precision, compile_args)
            
            logging.info(f"[MultiGPU] WanVideo VAE loaded on {selected_device}")
            return result
        else:
            logging.error(f"[MultiGPU] Could not patch WanVideo VAE modules")
            return original_loader.loadmodel(model_name, precision, compile_args)


class LoadWanVideoT5TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        from . import get_device_list
        devices = get_device_list()
        
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"),
                               {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
                "precision": (["fp32", "bf16"], {"default": "bf16"}),
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0], 
                                    "tooltip": "Device to load the text encoder to"}),
            },
            "optional": {
                "quantization": (['disabled', 'fp8_e4m3fn'],
                                 {"default": 'disabled', "tooltip": "optional quantization method"}),
            }
        }

    RETURN_TYPES = ("WANTEXTENCODER",)
    RETURN_NAMES = ("wan_t5_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan text_encoder model from 'ComfyUI/models/text_encoders'"

    def loadmodel(self, model_name, precision, device, quantization="disabled"):
        logging.debug(f"[MultiGPU] LoadWanVideoT5TextEncoder: User selected device: {device}")
        
        selected_device = torch.device(device)
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.debug(f"[MultiGPU] Patching WanVideo T5 modules to use {selected_device}")
            
            setattr(loader_module, 'device', selected_device)
            if device == "cpu":
                setattr(loader_module, 'offload_device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                setattr(nodes_module, 'device', selected_device)
                if device == "cpu":
                    setattr(nodes_module, 'offload_device', selected_device)
            
            result = original_loader.loadmodel(model_name, precision, load_device, quantization)
            
            logging.info(f"[MultiGPU] WanVideo T5 Text encoder loaded on {selected_device}")
            
            return result
        else:
            logging.error(f"[MultiGPU] Could not patch WanVideo T5 modules, falling back")
            return original_loader.loadmodel(model_name, precision, load_device, quantization)

class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        from . import get_device_list
        devices = get_device_list()
        
        return {"required": {
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0],
                                "tooltip": "Device to run the text encoding on"}),
            },
            "optional": {
                "t5": ("WANTEXTENCODER",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
                "use_disk_cache": ("BOOLEAN", {"default": False, "tooltip": "Cache the text embeddings to disk for faster re-use"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts with explicit device selection"
    
    def process(self, positive_prompt, negative_prompt, device, t5=None, force_offload=True, 
                model_to_offload=None, use_disk_cache=False):
        logging.debug(f"[MultiGPU] WanVideoTextEncode: User selected device: {device}")
        
        original_device = "gpu" if device != "cpu" else "cpu"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        
        encoder_module = inspect.getmodule(original_encoder)
        
        if encoder_module:
            selected_device = torch.device(device)
            logging.debug(f"[MultiGPU] Patching WanVideo TextEncode module to use {selected_device}")
            setattr(encoder_module, 'device', selected_device)
            
            model_loading_name = encoder_module.__name__.replace('.nodes', '.nodes_model_loading')
            if model_loading_name in sys.modules:
                model_loading_module = sys.modules[model_loading_name]
                setattr(model_loading_module, 'device', selected_device)
            
            result = original_encoder.process(positive_prompt, negative_prompt, t5=t5,
                                             force_offload=force_offload, model_to_offload=model_to_offload,
                                             use_disk_cache=use_disk_cache, device=original_device)
            
            logging.info(f"[MultiGPU] WanVideo TextEncode completed on {selected_device}")
            return result
        else:
            return original_encoder.process(positive_prompt, negative_prompt, t5=t5,
                                           force_offload=force_offload, model_to_offload=model_to_offload,
                                           use_disk_cache=use_disk_cache, device=original_device)

class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        from . import get_device_list
        devices = get_device_list()
        
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("clip_vision") + folder_paths.get_filename_list("text_encoders"),
                               {"tooltip": "These models are loaded from 'ComfyUI/models/clip_vision'"}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
                "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0],
                                    "tooltip": "Device to load the CLIP encoder to"}),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("clip_vision", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan CLIP text encoder model from 'ComfyUI/models/clip_vision'"

    def loadmodel(self, model_name, precision, device):
        logging.debug(f"[MultiGPU] LoadWanVideoClipTextEncoder: User selected device: {device}")
        
        selected_device = torch.device(device)
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoClipTextEncoder"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.debug(f"[MultiGPU] Patching WanVideo CLIP modules to use {selected_device}")
            
            setattr(loader_module, 'device', selected_device)
            if device == "cpu":
                setattr(loader_module, 'offload_device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                setattr(nodes_module, 'device', selected_device)
                if device == "cpu":
                    setattr(nodes_module, 'offload_device', selected_device)
            
            result = original_loader.loadmodel(model_name, precision, load_device)
            
            logging.info(f"[MultiGPU] WanVideo CLIP encoder loaded on {selected_device}")
            
            return result
        else:
            logging.error(f"[MultiGPU] Could not patch WanVideo CLIP modules, falling back")
            return original_loader.loadmodel(model_name, precision, load_device)

class WanVideoModelLoader_2:
    @classmethod
    def INPUT_TYPES(s):
        return WanVideoModelLoader.INPUT_TYPES()
    
    RETURN_TYPES = WanVideoModelLoader.RETURN_TYPES
    RETURN_NAMES = WanVideoModelLoader.RETURN_NAMES
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Second model loader instance for workflows using multiple models on different devices"
    
    def loadmodel(self, model, base_precision, device, quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, 
                  vram_management_args=None, vace_model=None, fantasytalking_model=None, multitalk_model=None):
        loader = WanVideoModelLoader()
        return loader.loadmodel(model, base_precision, device, quantization,
                              compile_args, attention_mode, block_swap_args, lora,
                              vram_management_args, vace_model, fantasytalking_model, multitalk_model)

class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        from nodes import NODE_CLASS_MAPPINGS
        original_types = NODE_CLASS_MAPPINGS["WanVideoSampler"].INPUT_TYPES()
        return original_types
    
    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware sampler that ensures correct device for each model"
    
    def process(self, model, **kwargs):
        model_device = model.load_device
        logging.info(f"[MultiGPU] WanVideoSampler: Processing on device: {model_device}")
        
        for module_name in sys.modules.keys():
            if 'WanVideoWrapper' in module_name and hasattr(sys.modules[module_name], 'device'):
                sys.modules[module_name].device = model_device
        
        from nodes import NODE_CLASS_MAPPINGS
        original_sampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        return original_sampler.process(model, **kwargs)

class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        from . import get_device_list
        devices = get_device_list()
        
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1, 
                                          "tooltip": "Number of transformer blocks to swap, the 14B model has 40, while the 1.3B model has 30 blocks"}),
                "swap_device": (devices, {"default": "cpu",
                                         "tooltip": "Device to swap blocks to during sampling (default: cpu for standard behavior)"}),
                "model_offload_device": (devices, {"default": "cpu",
                                                   "tooltip": "Device to offload entire model to when done (default: cpu)"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload img_emb to swap_device"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload time_emb to swap_device"}),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {"default": False, 
                                                  "tooltip": "Use non-blocking memory transfer for offloading, reserves more RAM but is faster"}),
                "vace_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1, 
                                               "tooltip": "Number of VACE blocks to swap, the VACE model has 15 blocks"}),
            },
        }
    
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Block swap settings with explicit device selection for memory management across GPUs"
    
    def setargs(self, blocks_to_swap, swap_device, model_offload_device, offload_img_emb, offload_txt_emb, 
                use_non_blocking=False, vace_blocks_to_swap=0):
        logging.debug(f"[MultiGPU] WanVideoBlockSwap: swap_device={swap_device}, model_offload_device={model_offload_device}, blocks_to_swap={blocks_to_swap}")
        
        selected_swap_device = torch.device(swap_device)
        selected_offload_device = torch.device(model_offload_device)
        
        for module_name in sys.modules.keys():
            if 'WanVideoWrapper' in module_name and 'nodes_model_loading' in module_name:
                module = sys.modules[module_name]
                setattr(module, 'offload_device', selected_offload_device)
                setattr(module, '_block_swap_device_override', selected_swap_device)
                setattr(module, '_model_offload_device_override', selected_offload_device)
                logging.debug(f"[MultiGPU] Patched {module_name} for offload to {selected_offload_device} and swap to {selected_swap_device}")

            if 'WanVideoWrapper' in module_name and module_name.endswith('.nodes'):
                module = sys.modules[module_name]
                setattr(module, 'offload_device', selected_offload_device)
                setattr(module, '_block_swap_device_override', selected_swap_device)
                setattr(module, '_model_offload_device_override', selected_offload_device)

        block_swap_args = {
            "blocks_to_swap": blocks_to_swap,
            "offload_img_emb": offload_img_emb,
            "offload_txt_emb": offload_txt_emb,
            "use_non_blocking": use_non_blocking,
            "vace_blocks_to_swap": vace_blocks_to_swap,
            "swap_device": swap_device,
            "model_offload_device": model_offload_device,
        }
        
        logging.info(f"[MultiGPU] WanVideoBlockSwap configuration complete")
        
        return (block_swap_args,)
