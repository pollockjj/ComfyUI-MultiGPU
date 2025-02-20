import os
import torch
import folder_paths
from pathlib import Path
from nodes import NODE_CLASS_MAPPINGS

class FluxFineTuneQuantizeAndLoad:
    @classmethod
    def INPUT_TYPES(cls):
        base_unet_name = folder_paths.get_filename_list("diffusion_models")
        unet_name = folder_paths.get_filename_list("diffusion_models")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "unet_name": (unet_name,),
                "switch_1": (["Off", "On"],),
                "lora_name_1": (loras,),
                "lora_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_2": (["Off", "On"],),
                "lora_name_2": (loras,),
                "lora_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_3": (["Off", "On"],),
                "lora_name_3": (loras,),
                "lora_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "switch_4": (["Off", "On"],),
                "lora_name_4": (loras,),
                "lora_weight_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "quantization": (["Q2_K", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_S", "Q6_K", "Q8_0", "FP16"], {"default": "Q4_K_S"}),
                "delete_final_gguf": ("BOOLEAN", {"default": False}),
                "new_model_name": ("STRING", {"default": "merged_model"}),
            }
        }
        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_and_quantize"
    CATEGORY = "loaders"

    def merge_flux_loras(self, model_sd: dict, lora_paths: list, weights: list, device="cuda") -> dict:
        for lora_path, weight in zip(lora_paths, weights):
            logging.info(f"[DEBUG] Merging LoRA file: {lora_path} with weight: {weight}")
            lora_sd = load_file(lora_path, device=device)
            for key in list(lora_sd.keys()):
                if "lora_down" not in key:
                    continue
                base_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                module_name = base_name.replace("_", ".")
                alpha_key = f"{base_name}.alpha"
                if module_name not in model_sd:
                    logging.info(f"[DEBUG] Module {module_name} not found in model_sd; skipping key {key}")
                    continue
                down_weight = lora_sd[key].float()
                up_weight = lora_sd[up_key].float()
                alpha = float(lora_sd.get(alpha_key, up_weight.shape[0]))
                scale = weight * alpha / up_weight.shape[0]
                logging.info(f"[DEBUG] Merging module: {module_name} with alpha: {alpha}, scale: {scale}")
                target_weight = model_sd[module_name]
                if len(target_weight.shape) == 2:
                    update = (up_weight @ down_weight) * scale
                else:
                    if down_weight.shape[2:4] == (1, 1):
                        update = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                        update = update.unsqueeze(2).unsqueeze(3) * scale
                    else:
                        update = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3) * scale
                model_sd[module_name] = target_weight + update.to(target_weight.dtype)
                logging.info(f"[DEBUG] Updated module: {module_name}")
                del up_weight, down_weight, update
            del lora_sd
            torch.cuda.empty_cache()
        return model_sd

    def convert_to_gguf(self, model_path, working_dir):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        convert_script = os.path.join(base_path, "ComfyUI-GGUF", "tools", "convert.py")
        temp_gguf = os.path.join(working_dir, "temp_converted.gguf")
        logging.info("[DEBUG] Running conversion script: " + convert_script)
        subprocess.run([sys.executable, convert_script, "--src", model_path, "--dst", temp_gguf], check=True)
        logging.info("[DEBUG] Conversion complete.")
        return temp_gguf

    def load_and_quantize(self, unet_name, quantization, delete_final_gguf, new_model_name, **kwargs):
        mapping = {"FP16": "F16"}
        logging.info(f"[DEBUG] Starting load_and_quantize: {new_model_name} | Quantization: {quantization}")
        with tempfile.TemporaryDirectory() as merge_dir:
            merged_model_path = os.path.join(merge_dir, "merged_model.safetensors")
            model_path = folder_paths.get_full_path("diffusion_models", unet_name)
            lora_list = []
            for i in range(1, 5):
                name = kwargs.get(f"lora_name_{i}", "None")
                switch = kwargs.get(f"switch_{i}", "Off")
                logging.info(f"[DEBUG] Processing LoRA slot {i}: name = {name}, switch = {switch}")
                if switch == "On" and name and name != "None":
                    lora_file_path = folder_paths.get_full_path("loras", name)
                    weight = kwargs.get(f"lora_weight_{i}", 1.0)
                    lora_list.append((lora_file_path, weight))
                    logging.info(f"[DEBUG] Slot {i} active: path = {lora_file_path}, weight = {weight}")
                else:
                    logging.info(f"[DEBUG] Slot {i} is inactive")
            logging.info(f"[DEBUG] Total active LoRAs: {len(lora_list)}")
            if lora_list:
                model_sd = load_file(model_path, device="cuda")
                model_sd = self.merge_flux_loras(
                    model_sd,
                    [lp for lp, _ in lora_list],
                    [w for _, w in lora_list]
                )
                save_file(model_sd, merged_model_path)
                del model_sd
                torch.cuda.empty_cache()
            else:
                shutil.copy2(model_path, merged_model_path)
            initial_gguf = self.convert_to_gguf(merged_model_path, merge_dir)
            logging.info("[DEBUG] Initial GGUF file created.")
            if quantization == "FP16":
                final_gguf = os.path.join(merge_dir, f"{new_model_name}-{mapping.get(quantization, quantization)}.gguf")
                shutil.copy2(initial_gguf, final_gguf)
                logging.info("[DEBUG] FP16 selected; conversion skipped.")
            else:
                binary = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binaries", "linux", "llama-quantize")
                final_gguf = os.path.join(merge_dir, f"quantized_{quantization}.gguf")
                subprocess.run([binary, initial_gguf, final_gguf, quantization], check=True)
                logging.info("[DEBUG] Quantization completed.")
            models_dir = os.path.join(folder_paths.models_dir, "unet")
            os.makedirs(models_dir, exist_ok=True)
            final_name = f"{new_model_name}-{mapping.get(quantization, quantization)}.gguf"
            final_path = os.path.join(models_dir, final_name)
            shutil.copy2(final_gguf, final_path)
            logging.info("[DEBUG] Final model file copied to: " + final_path)
            logging.info("[DEBUG] Loading final model.")
            loader = UnetLoaderGGUF()
            result = loader.load_unet(final_name)
            logging.info("[DEBUG] Final model loaded.")
            if delete_final_gguf:
                os.unlink(final_path)
            return result