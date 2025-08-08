import folder_paths
from pathlib import Path
from nodes import NODE_CLASS_MAPPINGS

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        return original_loader.load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"


class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        import nodes
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_data(ckpt_paths)

    def load_patcher(self, clip_paths, clip_type, clip_data):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_patcher(clip_paths, clip_type, clip_data)

    def load_clip(self, clip_name, type="stable_diffusion"):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_clip(clip_name, type)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        import nodes
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
        clip = original_loader.load_clip(clip_name1, clip_name2, type)
        clip[0].patcher.load(force_patch_weights=True)
        return clip


class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["TripleCLIPLoaderGGUF"]()
        return original_loader.load_clip(clip_name1, clip_name2, clip_name3, type)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
        }
    }

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["QuadrupleCLIPLoaderGGUF"]()
        return original_loader.load_clip(clip_name1, clip_name2, clip_name3, clip_name4, type)


class LTXVLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),
                            {"tooltip": "The name of the checkpoint (model) to load."}),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"})
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Loader"
    OUTPUT_NODE = False

    def load(self, ckpt_name, dtype):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
        return original_loader.load(ckpt_name, dtype)
    def _load_unet(self, load_device, offload_device, weights, num_latent_channels, dtype, config=None ):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
        return original_loader._load_unet(load_device, offload_device, weights, num_latent_channels, dtype, config=None )
    def _load_vae(self, weights, config=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
        return original_loader._load_vae(weights, config=None)

class Florence2ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ([item.name for item in Path(folder_paths.models_dir, "LLM").iterdir() if item.is_dir()], {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
            "precision": (['fp16','bf16','fp32'],),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["Florence2ModelLoader"]()
        return original_loader.loadmodel(model, precision, attention, lora)

class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'HuggingFaceM4/Florence-2-DocVQA',
                    'thwri/CogFlorence-2.1-Large',
                    'thwri/CogFlorence-2.2-Large',
                    'gokaygokay/Florence-2-SD3-Captioner',
                    'gokaygokay/Florence-2-Flux-Large',
                    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
                    'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
                    ],
                    {
                    "default": 'microsoft/Florence-2-base'
                    }),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"]()
        return original_loader.loadmodel(model, precision, attention, lora)

class CheckpointLoaderNF4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                            }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"


    def load_checkpoint(self, ckpt_name):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["CheckpointLoaderNF4"]()
        return original_loader.load_checkpoint(ckpt_name)

class LoadFluxControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (["flux-dev", "flux-dev-fp8", "flux-schnell"],),
                            "controlnet_path": (folder_paths.get_filename_list("xlabs_controlnets"), ),
                            }}

    RETURN_TYPES = ("FluxControlNet",)
    RETURN_NAMES = ("ControlNet",)
    FUNCTION = "loadmodel"
    CATEGORY = "XLabsNodes"

    def loadmodel(self, model_name, controlnet_path):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadFluxControlNet"]()
        return original_loader.loadmodel(model_name, controlnet_path)

class MMAudioModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from the 'ComfyUI/models/mmaudio' -folder",}),

            "base_precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MMAUDIO_MODEL",)
    RETURN_NAMES = ("mmaudio_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, mmaudio_model, base_precision):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["MMAudioModelLoader"]()
        return original_loader.loadmodel(mmaudio_model, base_precision)

class MMAudioFeatureUtilsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "synchformer_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "clip_model": (folder_paths.get_filename_list("mmaudio"), {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
            },
            "optional": {
            "bigvgan_vocoder_model": ("VOCODER_MODEL", {"tooltip": "These models are loaded from 'ComfyUI/models/mmaudio'"}),
                "mode": (["16k", "44k"], {"default": "44k"}),
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            }
        }

    RETURN_TYPES = ("MMAUDIO_FEATUREUTILS",)
    RETURN_NAMES = ("mmaudio_featureutils", )
    FUNCTION = "loadmodel"
    CATEGORY = "MMAudio"

    def loadmodel(self, vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["MMAudioFeatureUtilsLoader"]()
        return original_loader.loadmodel(vae_model, precision, synchformer_model, clip_model, mode, bigvgan_vocoder_model)

class MMAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mmaudio_model": ("MMAUDIO_MODEL",),
                "feature_utils": ("MMAUDIO_FEATUREUTILS",),
                "duration": ("FLOAT", {"default": 8, "step": 0.01, "tooltip": "Duration of the audio in seconds"}),
                "steps": ("INT", {"default": 25, "step": 1, "tooltip": "Number of steps to interpolate"}),
                "cfg": ("FLOAT", {"default": 4.5, "step": 0.1, "tooltip": "Strength of the conditioning"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True} ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
                "mask_away_clip": ("BOOLEAN", {"default": False, "tooltip": "If true, the clip video will be masked away"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "If true, the model will be offloaded to the offload device"}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio", )
    FUNCTION = "sample"
    CATEGORY = "MMAudio"

    def sample(self, mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip, force_offload, images=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["MMAudioSampler"]()
        return original_loader.sample(mmaudio_model, seed, feature_utils, duration, steps, cfg, prompt, negative_prompt, mask_away_clip, force_offload, images)

class PulidModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULID",)
    FUNCTION = "load_model"
    CATEGORY = "pulid"

    def load_model(self, pulid_file):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["PulidModelLoader"]()
        return original_loader.load_model(pulid_file)

class PulidInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "pulid"

    def load_insightface(self, provider):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["PulidInsightFaceLoader"]()
        return original_loader.load_insightface(provider)

class PulidEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "pulid"

    def load_eva_clip(self):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]()
        return original_loader.load_eva_clip()


class HyVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
                "base_precision": (["fp32", "bf16"], {"default": "bf16"}),
                "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_scaled', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], {"default": 'disabled', "tooltip": "optional quantization method"}),
                "load_device": (["main_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_varlen",
                    "sageattn_varlen",
                    "comfy",
                ], {"default": "flash_attn"}),
                "compile_args": ("COMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("HYVIDLORA", {"default": None}),
                "auto_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enable auto offloading for reduced VRAM usage, implementation from DiffSynth-Studio, slightly different from block swapping and uses even less VRAM, but can be slower as you can't define how much VRAM to use"}),
            }
        }

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device, quantization, compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, auto_cpu_offload=False):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["HyVideoModelLoader"]()
        return original_loader.loadmodel(model, base_precision, load_device, quantization, compile_args, attention_mode, block_swap_args, lora, auto_cpu_offload)

class HyVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
                "compile_args":("COMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["HyVideoVAELoader"]()
        return original_loader.loadmodel(model_name, precision, compile_args)

class DownloadAndLoadHyVideoTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_model": (["Kijai/llava-llama-3-8b-text-encoder-tokenizer","xtuner/llava-llama-3-8b-v1_1-transformers"],),
                "clip_model": (["disabled","openai/clip-vit-large-patch14",],),
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "apply_final_norm": ("BOOLEAN", {"default": False}),
                "hidden_state_skip_layer": ("INT", {"default": 2}),
                "quantization": (['disabled', 'bnb_nf4', "fp8_e4m3fn"], {"default": 'disabled'}),
            }
        }

    RETURN_TYPES = ("HYVIDTEXTENCODER",)
    RETURN_NAMES = ("hyvid_text_encoder", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, llm_model, clip_model, precision,  apply_final_norm=False, hidden_state_skip_layer=2, quantization="disabled"):
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadHyVideoTextEncoder"]()
        return original_loader.loadmodel(llm_model, clip_model, precision, apply_final_norm, hidden_state_skip_layer, quantization)
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
        import logging
        import comfy.model_management as mm
        import torch
        
        logging.info(f"[MultiGPU WanVideoModelLoader] ========== CUSTOM IMPLEMENTATION ==========")
        logging.info(f"[MultiGPU WanVideoModelLoader] User selected device: {device}")
        
        selected_device = torch.device(device)
        
        # Determine load_device for original loader
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        
        import sys
        import inspect
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.info(f"[MultiGPU WanVideoModelLoader] Patching WanVideo modules to use {selected_device}")
            
            original_device = getattr(loader_module, 'device', None)
            original_offload = getattr(loader_module, 'offload_device', None)
            
            # Check if there's a model offload device override (from block swap config)
            model_offload_override = getattr(loader_module, '_model_offload_device_override', None)
            
            setattr(loader_module, 'device', selected_device)
            if model_offload_override:
                setattr(loader_module, 'offload_device', model_offload_override)
                logging.info(f"[MultiGPU WanVideoModelLoader] Using model offload override: {model_offload_override}")
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
                logging.info(f"[MultiGPU WanVideoModelLoader] Both modules patched successfully")
            
            logging.info(f"[MultiGPU WanVideoModelLoader] Calling original loader")
            result = original_loader.loadmodel(model, base_precision, load_device, quantization,
                                              compile_args, attention_mode, block_swap_args, lora, vram_management_args, vace_model, fantasytalking_model, multitalk_model)
            
            # After model is loaded, check if we have a transformer and patch it for block swap
            if result and len(result) > 0 and hasattr(result[0], 'model'):
                model_obj = result[0]
                if hasattr(model_obj.model, 'diffusion_model'):
                    transformer = model_obj.model.diffusion_model
                    
                    block_swap_override = getattr(loader_module, '_block_swap_device_override', None)
                    if block_swap_override:
                        transformer.offload_device = block_swap_override
                        logging.info(f"[MultiGPU WanVideoModelLoader] Patched transformer for block swap to use: {block_swap_override}")
                    
            logging.info(f"[MultiGPU WanVideoModelLoader] Model loaded on {selected_device}")
            
            return result
        else:
            logging.error(f"[MultiGPU WanVideoModelLoader] Could not patch modules, falling back")
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
        import logging
        import torch
        
        logging.info(f"[MultiGPU WanVideoVAELoader] User selected device: {device}")
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        
        import sys
        import inspect
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            selected_device = torch.device(device)
            logging.info(f"[MultiGPU WanVideoVAELoader] Patching modules to use {selected_device}")
            
            setattr(loader_module, 'offload_device', selected_device)
            setattr(loader_module, 'device', selected_device)
            
            nodes_module_name = loader_module.__name__.replace('.nodes_model_loading', '.nodes')
            if nodes_module_name in sys.modules:
                nodes_module = sys.modules[nodes_module_name]
                setattr(nodes_module, 'device', selected_device)
                setattr(nodes_module, 'offload_device', selected_device)
            
            result = original_loader.loadmodel(model_name, precision, compile_args)
            
            logging.info(f"[MultiGPU WanVideoVAELoader] VAE loaded on {selected_device}")
            return result
        else:
            logging.error(f"[MultiGPU WanVideoVAELoader] Could not patch modules")
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
        import logging
        import torch
        
        logging.info(f"[MultiGPU LoadWanVideoT5TextEncoder] ========== CUSTOM IMPLEMENTATION ==========")
        logging.info(f"[MultiGPU LoadWanVideoT5TextEncoder] User selected device: {device}")
        
        selected_device = torch.device(device)
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        
        import sys
        import inspect
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.info(f"[MultiGPU LoadWanVideoT5TextEncoder] Patching WanVideo modules to use {selected_device}")
            
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
            
            logging.info(f"[MultiGPU LoadWanVideoT5TextEncoder] Text encoder loaded on {selected_device}")
            
            return result
        else:
            logging.error(f"[MultiGPU LoadWanVideoT5TextEncoder] Could not patch modules, falling back")
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
        import logging
        import torch
        
        logging.info(f"[MultiGPU WanVideoTextEncode] User selected device: {device}")
        
        original_device = "gpu" if device != "cpu" else "cpu"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_encoder = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        
        import sys
        import inspect
        encoder_module = inspect.getmodule(original_encoder)
        
        if encoder_module:
            selected_device = torch.device(device)
            logging.info(f"[MultiGPU WanVideoTextEncode] Patching module to use {selected_device}")
            setattr(encoder_module, 'device', selected_device)
            
            model_loading_name = encoder_module.__name__.replace('.nodes', '.nodes_model_loading')
            if model_loading_name in sys.modules:
                model_loading_module = sys.modules[model_loading_name]
                setattr(model_loading_module, 'device', selected_device)
            
            result = original_encoder.process(positive_prompt, negative_prompt, t5=t5,
                                             force_offload=force_offload, model_to_offload=model_to_offload,
                                             use_disk_cache=use_disk_cache, device=original_device)
            
            logging.info(f"[MultiGPU WanVideoTextEncode] Encoding completed on {selected_device}")
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
        import logging
        import torch
        
        logging.info(f"[MultiGPU LoadWanVideoClipTextEncoder] ========== CUSTOM IMPLEMENTATION ==========")
        logging.info(f"[MultiGPU LoadWanVideoClipTextEncoder] User selected device: {device}")
        
        selected_device = torch.device(device)
        load_device = "offload_device" if device == "cpu" else "main_device"
        
        from nodes import NODE_CLASS_MAPPINGS
        original_loader = NODE_CLASS_MAPPINGS["LoadWanVideoClipTextEncoder"]()
        
        import sys
        import inspect
        loader_module = inspect.getmodule(original_loader)
        
        if loader_module:
            logging.info(f"[MultiGPU LoadWanVideoClipTextEncoder] Patching WanVideo modules to use {selected_device}")
            
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
            
            logging.info(f"[MultiGPU LoadWanVideoClipTextEncoder] CLIP encoder loaded on {selected_device}")
            
            return result
        else:
            logging.error(f"[MultiGPU LoadWanVideoClipTextEncoder] Could not patch modules, falling back")
            return original_loader.loadmodel(model_name, precision, load_device)



class WanVideoModelLoader_2:
    """Second instance for multi-model workflows to maintain separate device patches"""
    @classmethod
    def INPUT_TYPES(s):
        # Delegate to the primary loader
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
    """Wrapper that ensures correct device patching before sampling"""
    @classmethod
    def INPUT_TYPES(s):
        # Get original sampler's inputs
        from nodes import NODE_CLASS_MAPPINGS
        original_types = NODE_CLASS_MAPPINGS["WanVideoSampler"].INPUT_TYPES()
        return original_types
    
    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("samples", "denoised_samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "MultiGPU-aware sampler that ensures correct device for each model"
    
    def process(self, model, **kwargs):
        import sys
        import torch
        import logging
        
        model_device = model.load_device
        logging.info(f"[MultiGPU WanVideoSampler] Processing on device: {model_device}")
        
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
        import logging
        import torch
        import comfy.model_management as mm
        
        logging.info(f"[MultiGPU WanVideoBlockSwap] ========== CONFIGURATION ==========")
        logging.info(f"[MultiGPU WanVideoBlockSwap] User selected swap device: {swap_device}")
        logging.info(f"[MultiGPU WanVideoBlockSwap] User selected model offload device: {model_offload_device}")
        logging.info(f"[MultiGPU WanVideoBlockSwap] Blocks to swap: {blocks_to_swap}")
        
        selected_swap_device = torch.device(swap_device)
        selected_offload_device = torch.device(model_offload_device)
        
        import sys
        
        for module_name in sys.modules.keys():
            if 'WanVideoWrapper' in module_name and 'nodes_model_loading' in module_name:
                module = sys.modules[module_name]
                setattr(module, 'offload_device', selected_offload_device)
                setattr(module, '_block_swap_device_override', selected_swap_device)
                setattr(module, '_model_offload_device_override', selected_offload_device)
                logging.info(f"[MultiGPU WanVideoBlockSwap] Patched {module_name} for offload to {selected_offload_device} and swap to {selected_swap_device}")

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
        
        logging.info(f"[MultiGPU WanVideoBlockSwap] Block swap configuration complete")
        
        return (block_swap_args,)
