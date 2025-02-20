import os
import torch
import folder_paths
import comfy.model_detection
import comfy.utils
import gguf
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
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv"],),
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
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": (("sdxl", "sd3", "flux", "hunyuan_video"),),
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
    
class ConverterFP16GGUF:
    QUANTIZATION_THRESHOLD = 1024
    REARRANGE_THRESHOLD = 512
    MAX_TENSOR_NAME_LENGTH = 127
    BLACKLIST = {"time_embedding.", "add_embedding.", "time_in.", "txt_in.", "vector_in.", "img_in.", "guidance_in.","final_layer.",}

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "safetensors_model": (s.get_diffusion_models_list(),),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "multigpu"
    FUNCTION = "execute"

    @classmethod
    def get_diffusion_models_list(s, folder_name="diffusion_models"):
        files = folder_paths.get_filename_list(folder_name)
        return sorted(files)

    def convert_to_fp16_gguf(self, safetensors_model):
        safetensors_path = folder_paths.get_full_path("diffusion_models", safetensors_model)
        base_filename = os.path.basename(os.path.splitext(safetensors_model)[0])
        output_name = f"{base_filename}-FP16"
        output_path = os.path.join(self.output_dir, f"{output_name}.gguf")

        try:
            state_dict = self.load_state_dict(safetensors_path)
            model_config, architecture_name = self.detect_architecture(state_dict)

            writer = gguf.GGUFWriter(path=None, arch=architecture_name)
            writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
            writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

            self.handle_tensors(writer, state_dict, output_path)

            writer.write_header_to_file(path=output_path)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            return output_path

        except Exception as e:
            return None

    def load_state_dict(self, path):
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
        temp_sd = comfy.utils.state_dict_prefix_replace(state_dict, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            state_dict = temp_sd
        return state_dict

    def detect_architecture(self, state_dict):
        model_config = comfy.model_detection.model_config_from_unet(state_dict, "")
        
        if model_config is None:
            raise ValueError("ComfyUI could not detect model architecture.")

        architecture_name = type(model_config).__name__
        return model_config, architecture_name

    def handle_tensors(self, writer, state_dict, output_path):
        name_lengths = tuple(sorted(((key, len(key)) for key in state_dict.keys()), key=lambda item: item[1], reverse=True))
        if not name_lengths:
            return
        max_name_len = name_lengths[0][1]
        if max_name_len > self.MAX_TENSOR_NAME_LENGTH:
            bad_list = ', '.join(f'{key!r} ({namelen})' for key, namelen in name_lengths if namelen > self.MAX_TENSOR_NAME_LENGTH)
            raise ValueError(f'Can only handle tensor names up to {self.MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}')
            
        for key, data in state_dict.items():
            old_dtype = data.dtype
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32).numpy()
            elif data.dtype in [getattr(torch, 'float8_e4m3fn', '_invalid'), getattr(torch, 'float8_e5m2', '_invalid')]:
                data = data.to(torch.float16).numpy()
            else:
                data = data.numpy()
            
            n_dims = len(data.shape)
            data_shape = data.shape
            data_qtype = getattr(gguf.GGMLQuantizationType, 'BF16' if old_dtype == torch.bfloat16 else 'F16')
            
            n_params = 1
            for dim_size in data_shape:
                n_params *= dim_size
                
            if old_dtype in (torch.float32, torch.bfloat16):
                if n_dims == 1:
                    data_qtype = gguf.GGMLQuantizationType.F32
                elif n_params <= self.QUANTIZATION_THRESHOLD:
                    data_qtype = gguf.GGMLQuantizationType.F32
                elif '.weight' in key and any(x in key for x in self.BLACKLIST):
                    data_qtype = gguf.GGMLQuantizationType.F32
                    
            try:
                data = gguf.quants.quantize(data, data_qtype)
            except (AttributeError, gguf.QuantError):
                data_qtype = gguf.GGMLQuantizationType.F16
                data = gguf.quants.quantize(data, data_qtype)
                
            writer.add_tensor(key, data, raw_dtype=data_qtype)

    def execute(self, safetensors_model):
        self.convert_to_fp16_gguf(safetensors_model)
        return ()