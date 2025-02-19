import os
import torch
import comfy.model_detection
import comfy.utils
import folder_paths
from tqdm import tqdm
import gguf
import logging

class ConverterComfy:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "safetensors_model": (s.get_diffusion_models_list(),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gguf_path",)
    OUTPUT_NODE = True
    CATEGORY = "multigpu"
    TITLE = "Converter Comfy (FP16 GGUF)"
    FUNCTION = "execute"

    @classmethod
    def get_diffusion_models_list(s, folder_name="diffusion_models"):
        files = folder_paths.get_filename_list(folder_name)
        return sorted(files)

    def convert_to_fp16_gguf(self, safetensors_model):
        safetensors_path = folder_paths.get_full_path("diffusion_models", safetensors_model)
        base_filename = os.path.splitext(safetensors_model)[0]
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
            writer.write_tensors_to_file(progress=True)
            writer.close()

            return output_path

        except Exception as e:
            error_message = f"Error during conversion: {e}"
            print(error_message)
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
            new_sd = comfy.model_detection.convert_diffusers_mmdit(state_dict, "")
            if new_sd is not None:
                model_config = comfy.model_detection.model_config_from_unet(new_sd, "")
                if model_config is None:
                    return self.diffusers_unet_fallback(state_dict)
            else:
                return self.diffusers_unet_fallback(state_dict)

        if model_config is None:
            raise ValueError("ComfyUI could not detect model architecture.")

        architecture_name = type(model_config).__name__
        print(f"* Architecture detected by ComfyUI: {architecture_name}")
        return model_config, architecture_name

    def diffusers_unet_fallback(self, state_dict):
        model_config = comfy.model_detection.model_config_from_diffusers_unet(state_dict)
        if model_config is None:
            return None

        diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

        new_sd = {}
        for k in diffusers_keys:
            if k in state_dict:
                new_sd[diffusers_keys[k]] = state_dict.pop(k)
            else:
                logging.warning(f"Key mapping warning: {diffusers_keys[k]} not found in state_dict as {k}")

        return model_config

    def handle_tensors(self, writer, state_dict, output_path):
        name_lengths = tuple(sorted(((key, len(key)) for key in state_dict.keys()), key=lambda item: item[1], reverse=True))
        if not name_lengths:
            return
        max_name_len = name_lengths[0][1]
        if max_name_len > 127:
            bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > 127)
            raise ValueError(f"Tensor names > 127 characters not supported. Tensors exceeding limit: {bad_list}")

        for key, data in tqdm(state_dict.items()):
            old_dtype = data.dtype
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float32).numpy()
            elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
                data = data.to(torch.float16).numpy()
            else:
                data = data.numpy()

            data_qtype = gguf.GGMLQuantizationType.F16

            shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
            tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{key}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
            writer.add_tensor(key, data, raw_dtype=data_qtype)


    def execute(self, safetensors_model):
        gguf_path = self.convert_to_fp16_gguf(safetensors_model)
        return (gguf_path, )


NODE_CLASS_MAPPINGS = {
    "ConverterComfy": ConverterComfy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConverterComfy": "Convert to FP16 GGUF (ComfyUI Detect)",
}