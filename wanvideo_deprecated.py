import logging
import torch
import sys
import inspect
import folder_paths
import comfy.model_management as mm
from .device_utils import get_device_list
from .model_management_mgpu import multigpu_memory_log

logger = logging.getLogger("MultiGPU")

class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
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
                "extra_model": ("VACEPATH", {"default": None, "tooltip": "Extra model to add to the main model, ie. VACE or MTV Crafter"}),
                "fantasytalking_model": ("FANTASYTALKMODEL", {"default": None, "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None, "tooltip": "Multitalk model"}),
                "fantasyportrait_model": ("FANTASYPORTRAITMODEL", {"default": None, "tooltip": "FantasyPortrait model"}),
                "rms_norm_function": (["default", "pytorch"], {"default": "default", "tooltip": "RMSNorm function to use, 'pytorch' is the new native torch RMSNorm, which is faster (when not using torch.compile mostly) but changes results slightly. 'default' is the original WanRMSNorm"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, device, quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None, extra_model=None, fantasytalking_model=None, multitalk_model=None, fantasyportrait_model=None, rms_norm_function="default"):
        from . import set_current_device
        
        logging.debug(f"[MultiGPU] WanVideoModelLoader: User selected device: {device}")
        
        selected_device = torch.device(device)
        
        # UPDATE GLOBAL DEVICE CONTEXT
        set_current_device(selected_device)
        
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
            
            logger.info(f"[MultiGPU WanVideo] Device patching complete. Calling original loader...")
            logger.info(f"[MultiGPU WanVideo] Module variables: device={loader_module.device}, offload_device={getattr(loader_module, 'offload_device', 'NOT SET')}")
            
            multigpu_memory_log("wanvideo_model_load", "pre-load")
            
            result = original_loader.loadmodel(model, base_precision, load_device, quantization,
                                              compile_args, attention_mode, block_swap_args, lora, vram_management_args, extra_model=extra_model, fantasytalking_model=fantasytalking_model, multitalk_model=multitalk_model, fantasyportrait_model=fantasyportrait_model, rms_norm_function=rms_norm_function)
            
            multigpu_memory_log("wanvideo_model_load", "post-load")
            
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
                                            compile_args, attention_mode, block_swap_args, lora, vram_management_args, extra_model=extra_model, fantasytalking_model=fantasytalking_model, multitalk_model=multitalk_model, fantasyportrait_model=fantasyportrait_model, rms_norm_function=rms_norm_function)


class WanVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
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
        from . import set_current_device
        
        logging.debug(f"[MultiGPU] WanVideoVAELoader: User selected device: {device}")
        
        selected_device = torch.device(device)
        
        # UPDATE GLOBAL DEVICE CONTEXT
        set_current_device(selected_device)
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.debug(f"[MultiGPU] Patching WanVideo VAE modules to use {selected_device}")
            
            setattr(loader_module, 'offload_device', selected_device)
            setattr(loader_module, 'device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                setattr(nodes_module, 'device', selected_device)
                setattr(nodes_module, 'offload_device', selected_device)
            
            multigpu_memory_log("wanvideo_vae_load", "pre-load")
            
            result = original_loader.loadmodel(model_name, precision, compile_args)
            
            multigpu_memory_log("wanvideo_vae_load", "post-load")
            
            # Attach device info to VAE object for downstream nodes
            if result and len(result) > 0:
                result[0].load_device = selected_device
            
            logger.info(f"[MultiGPU WanVideo VAE] VAE loaded successfully on {selected_device}")
            return result
        else:
            logging.error(f"[MultiGPU] Could not patch WanVideo VAE modules")
            return original_loader.loadmodel(model_name, precision, compile_args)


class LoadWanVideoT5TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
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
        import traceback
        from . import set_current_device
        
        logger.info(f"[T5 INSTRUMENT] ====== START LoadWanVideoT5TextEncoder ======")
        logger.info(f"[T5 INSTRUMENT] User selected device: {device}")
        logger.info(f"[T5 INSTRUMENT] Model name: {model_name}, Precision: {precision}, Quantization: {quantization}")
        
        selected_device = torch.device(device)
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            # PRE-PATCH STATE
            logger.info(f"[T5 INSTRUMENT] PRE-PATCH loader_module.device = {getattr(loader_module, 'device', 'NOT SET')}")
            logger.info(f"[T5 INSTRUMENT] PRE-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            # UPDATE GLOBAL DEVICE CONTEXT
            logger.info(f"[T5 INSTRUMENT] Calling set_current_device({selected_device})")
            set_current_device(selected_device)
            
            # WRAP mm.get_torch_device() TO LOG ALL CALLS
            original_mm_get_device = mm.get_torch_device
            call_count = [0]
            
            def logged_get_torch_device():
                call_count[0] += 1
                result = original_mm_get_device()
                stack = traceback.extract_stack()
                # Get caller info (skip this function and get the actual caller)
                caller = stack[-2] if len(stack) >= 2 else stack[-1]
                logger.info(f"[DEVICE CALL #{call_count[0]}] mm.get_torch_device() called from {caller.filename}:{caller.lineno} in {caller.name}() → returning {result}")
                return result
            
            mm.get_torch_device = logged_get_torch_device
            
            # WRAP MODULE DEVICE ACCESS
            original_device = getattr(loader_module, 'device', None)
            access_count = [0]
            
            class DeviceAccessLogger:
                def __init__(self, actual_device):
                    self._actual_device = actual_device
                
                def __str__(self):
                    access_count[0] += 1
                    stack = traceback.extract_stack()
                    caller = stack[-2] if len(stack) >= 2 else stack[-1]
                    logger.info(f"[MODULE ACCESS #{access_count[0]}] loader_module.device accessed from {caller.filename}:{caller.lineno} → returning {self._actual_device}")
                    return str(self._actual_device)
                
                def __repr__(self):
                    return str(self)
            
            # PATCH MODULE VARIABLES
            logger.info(f"[T5 INSTRUMENT] PATCHING loader_module.device to {selected_device}")
            setattr(loader_module, 'device', selected_device)
            if device == "cpu":
                setattr(loader_module, 'offload_device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                logger.info(f"[T5 INSTRUMENT] PATCHING nodes_module.device to {selected_device}")
                setattr(nodes_module, 'device', selected_device)
                if device == "cpu":
                    setattr(nodes_module, 'offload_device', selected_device)
            
            # POST-PATCH STATE
            logger.info(f"[T5 INSTRUMENT] POST-PATCH loader_module.device = {loader_module.device}")
            logger.info(f"[T5 INSTRUMENT] POST-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            multigpu_memory_log("wanvideo_t5_load", "pre-load")
            
            logger.info(f"[T5 INSTRUMENT] ===== CALLING ORIGINAL LOADER =====")
            logger.info(f"[T5 INSTRUMENT] Watch for DEVICE CALL and MODULE ACCESS logs below:")
            
            result = original_loader.loadmodel(model_name, precision, load_device, quantization)
            
            logger.info(f"[T5 INSTRUMENT] ===== ORIGINAL LOADER RETURNED =====")
            logger.info(f"[T5 INSTRUMENT] Total mm.get_torch_device() calls: {call_count[0]}")
            logger.info(f"[T5 INSTRUMENT] Total module.device accesses: {access_count[0]}")
            
            # RESTORE ORIGINAL
            mm.get_torch_device = original_mm_get_device
            
            multigpu_memory_log("wanvideo_t5_load", "post-load")
            
            # POST-LOAD STATE
            if result and len(result) > 0:
                t5_encoder = result[0]
                if isinstance(t5_encoder, dict) and 'model' in t5_encoder:
                    try:
                        actual_device = next(t5_encoder['model'].parameters()).device
                        logger.info(f"[T5 INSTRUMENT] ACTUAL model device after load = {actual_device}")
                    except Exception as e:
                        logger.info(f"[T5 INSTRUMENT] Could not determine actual model device: {e}")
            
            logger.info(f"[T5 INSTRUMENT] ====== END LoadWanVideoT5TextEncoder ======")
            
            return result
        else:
            logger.error(f"[T5 INSTRUMENT] Could not get loader module - falling back")
            return original_loader.loadmodel(model_name, precision, load_device, quantization)

class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
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
        import traceback
        from . import set_current_device
        
        logger.info(f"[TEXTENCODE INSTRUMENT] ====== START WanVideoTextEncode ======")
        logger.info(f"[TEXTENCODE INSTRUMENT] User selected device: {device}")
        
        selected_device = torch.device(device)
        original_device = "gpu" if device != "cpu" else "cpu"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        
        encoder_module = inspect.getmodule(original_encoder)
        
        if encoder_module:
            # PRE-PATCH STATE
            logger.info(f"[TEXTENCODE INSTRUMENT] PRE-PATCH encoder_module.device = {getattr(encoder_module, 'device', 'NOT SET')}")
            logger.info(f"[TEXTENCODE INSTRUMENT] PRE-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            # UPDATE GLOBAL DEVICE CONTEXT
            logger.info(f"[TEXTENCODE INSTRUMENT] Calling set_current_device({selected_device})")
            set_current_device(selected_device)
            
            # WRAP mm.get_torch_device() TO LOG ALL CALLS
            original_mm_get_device = mm.get_torch_device
            call_count = [0]
            
            def logged_get_torch_device():
                call_count[0] += 1
                result = original_mm_get_device()
                stack = traceback.extract_stack()
                caller = stack[-2] if len(stack) >= 2 else stack[-1]
                logger.info(f"[DEVICE CALL #{call_count[0]}] mm.get_torch_device() called from {caller.filename}:{caller.lineno} in {caller.name}() → returning {result}")
                return result
            
            mm.get_torch_device = logged_get_torch_device
            
            # PATCH MODULE VARIABLES
            logger.info(f"[TEXTENCODE INSTRUMENT] PATCHING encoder_module.device to {selected_device}")
            setattr(encoder_module, 'device', selected_device)
            
            model_loading_name = encoder_module.__name__.replace('.nodes', '.nodes_model_loading')
            if model_loading_name in sys.modules:
                model_loading_module = sys.modules[model_loading_name]
                logger.info(f"[TEXTENCODE INSTRUMENT] PATCHING model_loading_module.device to {selected_device}")
                setattr(model_loading_module, 'device', selected_device)
            
            # POST-PATCH STATE
            logger.info(f"[TEXTENCODE INSTRUMENT] POST-PATCH encoder_module.device = {encoder_module.device}")
            logger.info(f"[TEXTENCODE INSTRUMENT] POST-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            multigpu_memory_log("wanvideo_textencode", "pre-encode")
            
            logger.info(f"[TEXTENCODE INSTRUMENT] ===== CALLING ORIGINAL ENCODER =====")
            
            result = original_encoder.process(positive_prompt, negative_prompt, t5=t5,
                                             force_offload=force_offload, model_to_offload=model_to_offload,
                                             use_disk_cache=use_disk_cache, device=original_device)
            
            multigpu_memory_log("wanvideo_textencode", "post-encode")
            
            logger.info(f"[TEXTENCODE INSTRUMENT] ===== ORIGINAL ENCODER RETURNED =====")
            logger.info(f"[TEXTENCODE INSTRUMENT] Total mm.get_torch_device() calls: {call_count[0]}")
            
            # RESTORE ORIGINAL
            mm.get_torch_device = original_mm_get_device
            
            logger.info(f"[TEXTENCODE INSTRUMENT] ====== END WanVideoTextEncode ======")
            return result
        else:
            logger.error(f"[TEXTENCODE INSTRUMENT] Could not get encoder module - falling back")
            return original_encoder.process(positive_prompt, negative_prompt, t5=t5,
                                           force_offload=force_offload, model_to_offload=model_to_offload,
                                           use_disk_cache=use_disk_cache, device=original_device)

class WanVideoTextEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True}),
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

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes single text prompt with explicit device selection"
    
    def process(self, prompt, device, t5=None, force_offload=True, 
                model_to_offload=None, use_disk_cache=False):
        import traceback
        from . import set_current_device
        
        logger.info(f"[TEXTENCODESINGLE INSTRUMENT] ====== START WanVideoTextEncodeSingle ======")
        logger.info(f"[TEXTENCODESINGLE INSTRUMENT] User selected device: {device}")
        
        selected_device = torch.device(device)
        original_device = "gpu" if device != "cpu" else "cpu"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncodeSingle"]()
        
        encoder_module = inspect.getmodule(original_encoder)
        
        if encoder_module:
            # PRE-PATCH STATE
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] PRE-PATCH encoder_module.device = {getattr(encoder_module, 'device', 'NOT SET')}")
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] PRE-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            # UPDATE GLOBAL DEVICE CONTEXT
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] Calling set_current_device({selected_device})")
            set_current_device(selected_device)
            
            # WRAP mm.get_torch_device() TO LOG ALL CALLS
            original_mm_get_device = mm.get_torch_device
            call_count = [0]
            
            def logged_get_torch_device():
                call_count[0] += 1
                result = original_mm_get_device()
                stack = traceback.extract_stack()
                caller = stack[-2] if len(stack) >= 2 else stack[-1]
                logger.info(f"[DEVICE CALL #{call_count[0]}] mm.get_torch_device() called from {caller.filename}:{caller.lineno} in {caller.name}() → returning {result}")
                return result
            
            mm.get_torch_device = logged_get_torch_device
            
            # PATCH MODULE VARIABLES
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] PATCHING encoder_module.device to {selected_device}")
            setattr(encoder_module, 'device', selected_device)
            
            model_loading_name = encoder_module.__name__.replace('.nodes', '.nodes_model_loading')
            if model_loading_name in sys.modules:
                model_loading_module = sys.modules[model_loading_name]
                logger.info(f"[TEXTENCODESINGLE INSTRUMENT] PATCHING model_loading_module.device to {selected_device}")
                setattr(model_loading_module, 'device', selected_device)
            
            # POST-PATCH STATE
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] POST-PATCH encoder_module.device = {encoder_module.device}")
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] POST-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            multigpu_memory_log("wanvideo_textencodesingle", "pre-encode")
            
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] ===== CALLING ORIGINAL ENCODER =====")
            
            result = original_encoder.process(prompt, t5=t5,
                                             force_offload=force_offload, model_to_offload=model_to_offload,
                                             use_disk_cache=use_disk_cache, device=original_device)
            
            multigpu_memory_log("wanvideo_textencodesingle", "post-encode")
            
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] ===== ORIGINAL ENCODER RETURNED =====")
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] Total mm.get_torch_device() calls: {call_count[0]}")
            
            # RESTORE ORIGINAL
            mm.get_torch_device = original_mm_get_device
            
            logger.info(f"[TEXTENCODESINGLE INSTRUMENT] ====== END WanVideoTextEncodeSingle ======")
            return result
        else:
            logger.error(f"[TEXTENCODESINGLE INSTRUMENT] Could not get encoder module - falling back")
            return original_encoder.process(prompt, t5=t5,
                                           force_offload=force_offload, model_to_offload=model_to_offload,
                                           use_disk_cache=use_disk_cache, device=original_device)


class WanVideoTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        
        return {"required": {
            "model_name": (folder_paths.get_filename_list("text_encoders"),
                          {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
            "precision": (["fp32", "bf16"], {"default": "bf16"}),
            "positive_prompt": ("STRING", {"default": "", "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "quantization": (["disabled", "fp8_e4m3fn"], {"default": "disabled"}),
            "use_disk_cache": ("BOOLEAN", {"default": True}),
            "device": (devices, {"default": devices[1] if len(devices) > 1 else devices[0],
                                "tooltip": "Device to run the text encoding on"}),
            },
            "optional": {
                "extender_args": ("WANVIDEOPROMPTEXTENDER_ARGS",),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("text_embeds", "negative_text_embeds", "positive_prompt")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Cached text encoding with explicit device selection"
    
    def process(self, model_name, precision, positive_prompt, negative_prompt, 
                quantization, use_disk_cache, device, extender_args=None):
        import traceback
        from . import set_current_device
        
        logger.info(f"[TEXTENCODECACHED INSTRUMENT] ====== START WanVideoTextEncodeCached ======")
        logger.info(f"[TEXTENCODECACHED INSTRUMENT] User selected device: {device}")
        
        selected_device = torch.device(device)
        original_device = "gpu" if device != "cpu" else "cpu"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
        
        encoder_module = inspect.getmodule(original_encoder)
        
        if encoder_module:
            # PRE-PATCH STATE
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] PRE-PATCH encoder_module.device = {getattr(encoder_module, 'device', 'NOT SET')}")
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] PRE-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            # UPDATE GLOBAL DEVICE CONTEXT
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] Calling set_current_device({selected_device})")
            set_current_device(selected_device)
            
            # WRAP mm.get_torch_device() TO LOG ALL CALLS
            original_mm_get_device = mm.get_torch_device
            call_count = [0]
            
            def logged_get_torch_device():
                call_count[0] += 1
                result = original_mm_get_device()
                stack = traceback.extract_stack()
                caller = stack[-2] if len(stack) >= 2 else stack[-1]
                logger.info(f"[DEVICE CALL #{call_count[0]}] mm.get_torch_device() called from {caller.filename}:{caller.lineno} in {caller.name}() → returning {result}")
                return result
            
            mm.get_torch_device = logged_get_torch_device
            
            # PATCH MODULE VARIABLES
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] PATCHING encoder_module.device to {selected_device}")
            setattr(encoder_module, 'device', selected_device)
            
            model_loading_name = encoder_module.__name__.replace('.nodes', '.nodes_model_loading')
            if model_loading_name in sys.modules:
                model_loading_module = sys.modules[model_loading_name]
                logger.info(f"[TEXTENCODECACHED INSTRUMENT] PATCHING model_loading_module.device to {selected_device}")
                setattr(model_loading_module, 'device', selected_device)
            
            # POST-PATCH STATE
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] POST-PATCH encoder_module.device = {encoder_module.device}")
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] POST-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            multigpu_memory_log("wanvideo_textencodecached", "pre-encode")
            
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] ===== CALLING ORIGINAL ENCODER =====")
            
            result = original_encoder.process(model_name, precision, positive_prompt, negative_prompt,
                                             quantization, use_disk_cache, device=original_device,
                                             extender_args=extender_args)
            
            multigpu_memory_log("wanvideo_textencodecached", "post-encode")
            
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] ===== ORIGINAL ENCODER RETURNED =====")
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] Total mm.get_torch_device() calls: {call_count[0]}")
            
            # RESTORE ORIGINAL
            mm.get_torch_device = original_mm_get_device
            
            logger.info(f"[TEXTENCODECACHED INSTRUMENT] ====== END WanVideoTextEncodeCached ======")
            return result
        else:
            logger.error(f"[TEXTENCODECACHED INSTRUMENT] Could not get encoder module - falling back")
            return original_encoder.process(model_name, precision, positive_prompt, negative_prompt,
                                           quantization, use_disk_cache, device=original_device,
                                           extender_args=extender_args)


class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
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
        import traceback
        from . import set_current_device
        
        logger.info(f"[CLIP INSTRUMENT] ====== START LoadWanVideoClipTextEncoder ======")
        logger.info(f"[CLIP INSTRUMENT] User selected device: {device}")
        logger.info(f"[CLIP INSTRUMENT] Model name: {model_name}, Precision: {precision}")
        
        selected_device = torch.device(device)
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoClipTextEncoder"]()
        
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            # PRE-PATCH STATE
            logger.info(f"[CLIP INSTRUMENT] PRE-PATCH loader_module.device = {getattr(loader_module, 'device', 'NOT SET')}")
            logger.info(f"[CLIP INSTRUMENT] PRE-PATCH loader_module.offload_device = {getattr(loader_module, 'offload_device', 'NOT SET')}")
            logger.info(f"[CLIP INSTRUMENT] PRE-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            # UPDATE GLOBAL DEVICE CONTEXT
            logger.info(f"[CLIP INSTRUMENT] Calling set_current_device({selected_device})")
            set_current_device(selected_device)
            
            # WRAP mm.get_torch_device() TO LOG ALL CALLS
            original_mm_get_device = mm.get_torch_device
            call_count = [0]
            
            def logged_get_torch_device():
                call_count[0] += 1
                result = original_mm_get_device()
                stack = traceback.extract_stack()
                caller = stack[-2] if len(stack) >= 2 else stack[-1]
                logger.info(f"[DEVICE CALL #{call_count[0]}] mm.get_torch_device() called from {caller.filename}:{caller.lineno} in {caller.name}() → returning {result}")
                return result
            
            mm.get_torch_device = logged_get_torch_device
            
            # WRAP MODULE DEVICE ACCESS
            original_device = getattr(loader_module, 'device', None)
            access_count = [0]
            
            class DeviceAccessLogger:
                def __init__(self, actual_device):
                    self._actual_device = actual_device
                
                def __str__(self):
                    access_count[0] += 1
                    stack = traceback.extract_stack()
                    caller = stack[-2] if len(stack) >= 2 else stack[-1]
                    logger.info(f"[MODULE ACCESS #{access_count[0]}] loader_module.device accessed from {caller.filename}:{caller.lineno} → returning {self._actual_device}")
                    return str(self._actual_device)
                
                def __repr__(self):
                    return str(self)
            
            # PATCH MODULE VARIABLES
            logger.info(f"[CLIP INSTRUMENT] PATCHING loader_module.device to {selected_device}")
            setattr(loader_module, 'device', selected_device)
            if device == "cpu":
                setattr(loader_module, 'offload_device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                logger.info(f"[CLIP INSTRUMENT] PRE-PATCH nodes_module.device = {getattr(nodes_module, 'device', 'NOT SET')}")
                logger.info(f"[CLIP INSTRUMENT] PATCHING nodes_module.device to {selected_device}")
                setattr(nodes_module, 'device', selected_device)
                if device == "cpu":
                    setattr(nodes_module, 'offload_device', selected_device)
            
            # POST-PATCH STATE
            logger.info(f"[CLIP INSTRUMENT] POST-PATCH loader_module.device = {loader_module.device}")
            logger.info(f"[CLIP INSTRUMENT] POST-PATCH loader_module.offload_device = {getattr(loader_module, 'offload_device', 'NOT SET')}")
            logger.info(f"[CLIP INSTRUMENT] POST-PATCH mm.get_torch_device() = {mm.get_torch_device()}")
            
            multigpu_memory_log("wanvideo_clip_load", "pre-load")
            
            logger.info(f"[CLIP INSTRUMENT] ===== CALLING ORIGINAL LOADER =====")
            logger.info(f"[CLIP INSTRUMENT] Watch for DEVICE CALL and MODULE ACCESS logs below:")
            
            result = original_loader.loadmodel(model_name, precision, load_device)
            
            logger.info(f"[CLIP INSTRUMENT] ===== ORIGINAL LOADER RETURNED =====")
            logger.info(f"[CLIP INSTRUMENT] Total mm.get_torch_device() calls: {call_count[0]}")
            logger.info(f"[CLIP INSTRUMENT] Total module.device accesses: {access_count[0]}")
            
            # RESTORE ORIGINAL
            mm.get_torch_device = original_mm_get_device
            
            multigpu_memory_log("wanvideo_clip_load", "post-load")
            
            # POST-LOAD STATE
            if result and len(result) > 0:
                clip_model = result[0]
                if hasattr(clip_model, 'model'):
                    try:
                        actual_device = next(clip_model.model.parameters()).device
                        logger.info(f"[CLIP INSTRUMENT] ACTUAL model device after load = {actual_device}")
                    except Exception as e:
                        logger.info(f"[CLIP INSTRUMENT] Could not determine actual model device: {e}")
                logger.info(f"[CLIP INSTRUMENT] Result type: {type(clip_model)}")
            
            logger.info(f"[CLIP INSTRUMENT] ====== END LoadWanVideoClipTextEncoder ======")
            
            return result
        else:
            logger.error(f"[CLIP INSTRUMENT] Could not get loader module - falling back")
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
                  vram_management_args=None, vace_model=None, fantasytalking_model=None, multitalk_model=None, fantasyportrait_model=None, rms_norm_function="default"):
        loader = WanVideoModelLoader()
        return loader.loadmodel(model, base_precision, device, quantization,
                              compile_args, attention_mode, block_swap_args, lora,
                              vram_management_args, vace_model, fantasytalking_model, multitalk_model, fantasyportrait_model, rms_norm_function)

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

class WanVideoVACEEncode:
    @classmethod
    def INPUT_TYPES(s):
        from nodes import NODE_CLASS_MAPPINGS
        original_types = NODE_CLASS_MAPPINGS["WanVideoVACEEncode"].INPUT_TYPES()
        return original_types
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware VACE encoder that uses device from input VAE"
    
    def process(self, vae, **kwargs):
        # Get device from VAE object
        vae_device = vae.load_device
        logging.info(f"[MultiGPU] WanVideoVACEEncode: Processing on device: {vae_device}")
        
        # Patch all WanVideo modules to use the VAE's device
        for module_name in sys.modules.keys():
            if 'WanVideoWrapper' in module_name and hasattr(sys.modules[module_name], 'device'):
                sys.modules[module_name].device = vae_device
        
        from nodes import NODE_CLASS_MAPPINGS
        original_encoder = NODE_CLASS_MAPPINGS["WanVideoVACEEncode"]()
        return original_encoder.process(vae, **kwargs)

class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
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
                "prefetch_blocks": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1, "tooltip": "Number of blocks to prefetch ahead, can speed up processing but increases memory usage. 1 is usually enough to offset speed loss from block swapping, use the debug option to confirm it for your system"}),
                "block_swap_debug": ("BOOLEAN", {"default": False, "tooltip": "Enable debug logging for block swapping"}),
            },
        }
    
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Block swap settings with explicit device selection for memory management across GPUs"
    
    def setargs(self, blocks_to_swap, swap_device, model_offload_device, offload_img_emb, offload_txt_emb, 
                use_non_blocking=False, vace_blocks_to_swap=0, prefetch_blocks=0, block_swap_debug=False):
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
            "prefetch_blocks": prefetch_blocks,
            "block_swap_debug": block_swap_debug,
            "swap_device": swap_device,
            "model_offload_device": model_offload_device,
        }
        
        logging.info(f"[MultiGPU] WanVideoBlockSwap configuration complete")
        
        return (block_swap_args,)
