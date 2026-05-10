import folder_paths
from pathlib import Path
from nodes import NODE_CLASS_MAPPINGS
from .device_utils import get_device_list

class DeviceSelectorMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        devices = get_device_list()
        return {"required": {"device": (devices,)}}

    RETURN_TYPES = ("MULTIGPUDEVICE",)
    RETURN_NAMES = ("device",)
    FUNCTION = "select_device"
    CATEGORY = "multigpu"
    TITLE = "Device Selector (MultiGPU)"

    def select_device(self, device):
        """Return the selected device label without side effects."""
        return (device,)

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = list(folder_paths.get_filename_list("unet_gguf"))
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
        """Load GGUF format UNet model."""
        original_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        return original_loader.load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = list(folder_paths.get_filename_list("unet_gguf"))
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
        """Get combined list of CLIP and CLIP_GGUF model files."""
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        """Load CLIP model data from checkpoint paths."""
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_data(ckpt_paths)

    def load_patcher(self, clip_paths, clip_type, clip_data):
        """Create ModelPatcher for CLIP model."""
        original_loader = NODE_CLASS_MAPPINGS["CLIPLoaderGGUF"]()
        return original_loader.load_patcher(clip_paths, clip_type, clip_data)

    def load_clip(self, clip_name, type="stable_diffusion", device=None):
        """Load CLIP model from GGUF or standard format."""
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

    def load_clip(self, clip_name1, clip_name2, type, device=None):
        """Load dual CLIP model configuration."""
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
        """Load triple CLIP model configuration for SD3."""
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
        """Load quadruple CLIP model configuration."""
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
        """Load LTXV model and VAE with specified precision."""
        original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
        return original_loader.load(ckpt_name, dtype)
    def _load_unet(self, load_device, offload_device, weights, num_latent_channels, dtype, config=None ):
        """Load LTXV UNet with device-specific configuration."""
        original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
        return original_loader._load_unet(load_device, offload_device, weights, num_latent_channels, dtype, config=None )
    def _load_vae(self, weights, config=None):
        """Load LTXV VAE from weights."""
        original_loader = NODE_CLASS_MAPPINGS["LTXVLoader"]()
        return original_loader._load_vae(weights, config=None)

class Florence2ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        # Delegate to upstream so the input set tracks Florence2 verbatim
        # (e.g. post-rewrite removal of `attention` from required) and so
        # upstream's `model_paths` class attribute gets populated as a
        # side-effect — `loadmodel` reads it via `Florence2ModelLoader.model_paths.get(model)`.
        return NODE_CLASS_MAPPINGS["Florence2ModelLoader"].INPUT_TYPES()

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, *args, **kwargs):
        """Forward to the upstream Florence2 loader without re-binding args."""
        original_loader = NODE_CLASS_MAPPINGS["Florence2ModelLoader"]()
        return original_loader.loadmodel(*args, **kwargs)

class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        # Delegate to upstream so the model list and signature stay in sync
        # with kijai/ComfyUI-Florence2 (e.g. PromptGen v2.0 and Castollux
        # additions, attention parameter relocation).
        return NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"].INPUT_TYPES()

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, *args, **kwargs):
        """Forward to the upstream HuggingFace downloader/loader."""
        original_loader = NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"]()
        return original_loader.loadmodel(*args, **kwargs)

class CheckpointLoaderNF4:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                            }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"


    def load_checkpoint(self, ckpt_name):
        """Load checkpoint in NF4 quantized format."""
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
        """Load Flux ControlNet model."""
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
        """Load MMAudio model with specified precision."""
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
        """Load MMAudio feature extraction utilities including VAE, Synchformer, and CLIP."""
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
        """Sample audio from MMAudio model with conditioning."""
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
        """Load PuLID identity preservation model."""
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
        """Load InsightFace face analysis model for PuLID."""
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
        """Load EVA CLIP model for PuLID."""
        original_loader = NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]()
        return original_loader.load_eva_clip()

class UNetLoaderLP:
    """UNet Loader (Low Precision) - sets LoRA precision to False for CPU storage optimization"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("unet"), ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    TITLE = "UNet Loader (LP)"

    def load_unet(self, unet_name):
        """Load UNet with low-precision LoRA flag for CPU storage optimization."""
        original_loader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        out = original_loader.load_unet(unet_name, "default")

        # Set the low-precision LoRA flag on the loaded model
        if hasattr(out[0], 'model'):
            out[0].model._distorch_high_precision_loras = False
        else:
            patcher = getattr(out[0], "patcher", None)
            if patcher is not None and hasattr(patcher, "model"):
                patcher.model._distorch_high_precision_loras = False

        return out
