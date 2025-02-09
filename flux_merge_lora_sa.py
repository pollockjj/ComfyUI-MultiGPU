import argparse
import math
import os
import time
from typing import Any, Dict, Union, List, Optional, Tuple, Type

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import json
import struct
import re
import logging

# --- From utils.py ---
def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                with torch.cuda.device(v.device):
                    if v.dim() == 0:
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                if v.dim() == 0:
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    def __init__(self, filename):
        self.filename = filename
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start == offset_end:
            tensor_bytes = None
        else:
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_bytes = bytearray(tensor_bytes)
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


# --- From lora_flux.py ---
class LoRAModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        pass
    def forward(self, *args, **kwargs):
        pass

class LoRAInfModule(LoRAModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        pass
    def merge_to(self, *args, **kwargs):
        pass
    def get_weight(self, *args, **kwargs):
        pass

class LoRANetwork(torch.nn.Module):
    LORA_PREFIX_FLUX = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER_CLIP = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER_T5 = "lora_te3"
    def __init__(self, *args, **kwargs):
        super().__init__()
    def apply_to(self, *args, **kwargs):
        pass
    def is_mergeable(self, *args, **kwargs):
        return True
    def merge_to(self, *args, **kwargs):
        pass
    def load_state_dict(self, *args, **kwargs):
        pass
    def state_dict(self, *args, **kwargs):
        pass

# --- From flux_merge_lora.py ---
def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = {} # simplified, no metadata loading
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}
    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)
    return sd, metadata


def save_to_file(file_name, state_dict: Dict[str, Union[Any, torch.Tensor]], dtype, metadata, mem_eff_save=False):
    if dtype is not None:
        for key in tqdm(list(state_dict.keys())):
            if type(state_dict[key]) == torch.Tensor and state_dict[key].dtype.is_floating_point:
                state_dict[key] = state_dict[key].to(dtype)

    if mem_eff_save:
        mem_eff_save_file(state_dict, file_name, metadata=metadata)
    else:
        save_file(state_dict, file_name, metadata=metadata)


def merge_to_flux_model(
    loading_device,
    working_device,
    flux_path: str,
    clip_l_path: str,
    t5xxl_path: str,
    models,
    ratios,
    merge_dtype,
    save_dtype,
    mem_eff_load_save=False,
):
    lora_name_to_module_key = {}
    if flux_path is not None:
        with safe_open(flux_path, framework="pt", device=loading_device) as flux_file:
            keys = list(flux_file.keys())
            for key in keys:
                if key.endswith(".weight"):
                    module_name = ".".join(key.split(".")[:-1])
                    lora_name = LoRANetwork.LORA_PREFIX_FLUX + "_" + module_name.replace(".", "_")
                    lora_name_to_module_key[lora_name] = key

    lora_name_to_clip_l_key = {}
    lora_name_to_t5xxl_key = {}

    flux_state_dict = {}
    clip_l_state_dict = {}
    t5xxl_state_dict = {}
    if mem_eff_load_save:
        if flux_path is not None:
            with MemoryEfficientSafeOpen(flux_path) as flux_file:
                for key in tqdm(flux_file.keys()):
                    flux_state_dict[key] = flux_file.get_tensor(key).to(loading_device)

        if clip_l_path is not None:
            with MemoryEfficientSafeOpen(clip_l_path) as clip_l_file:
                for key in tqdm(clip_l_file.keys()):
                    clip_l_state_dict[key] = clip_l_file.get_tensor(key).to(loading_device)

        if t5xxl_path is not None:
            with MemoryEfficientSafeOpen(t5xxl_path) as t5xxl_file:
                for key in tqdm(t5xxl_file.keys()):
                    t5xxl_state_dict[key] = t5xxl_file.get_tensor(key).to(loading_device)
    else:
        if flux_path is not None:
            flux_state_dict = load_file(flux_path, device=loading_device)
        if clip_l_path is not None:
            clip_l_state_dict = load_file(clip_l_path, device=loading_device)
        if t5xxl_path is not None:
            t5xxl_state_dict = load_file(t5xxl_path, device=loading_device)

    for model, ratio in zip(models, ratios):
        lora_sd, _ = load_state_dict(model, merge_dtype)

        for key in tqdm(list(lora_sd.keys())):
            if "lora_down" in key:
                lora_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                if lora_name in lora_name_to_module_key:
                    module_weight_key = lora_name_to_module_key[lora_name]
                    state_dict = flux_state_dict
                elif lora_name in lora_name_to_clip_l_key:
                    module_weight_key = lora_name_to_clip_l_key[lora_name]
                    state_dict = clip_l_state_dict
                elif lora_name in lora_name_to_t5xxl_key:
                    module_weight_key = lora_name_to_t5xxl_key[lora_name]
                    state_dict = t5xxl_state_dict
                else:
                    continue

                down_weight = lora_sd.pop(key)
                up_weight = lora_sd.pop(up_key)

                dim = down_weight.size()[0]
                alpha = lora_sd.pop(alpha_key, dim)
                scale = alpha / dim

                weight = state_dict[module_weight_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                if len(weight.size()) == 2:
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    weight = weight + ratio * conved * scale

                state_dict[module_weight_key] = weight.to(loading_device, save_dtype)
                del up_weight
                del down_weight
                del weight

        if len(lora_sd) > 0:
            pass # unused keys warning removed for standalone version

    return flux_state_dict, clip_l_state_dict, t5xxl_state_dict


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_precision", type=str, default=None)
    parser.add_argument("--precision", type=str, default="float")
    parser.add_argument("--flux_model", type=str, default=None)
    parser.add_argument("--clip_l", type=str, default=None)
    parser.add_argument("--t5xxl", type=str, default=None)
    parser.add_argument("--mem_eff_load_save", action="store_true")
    parser.add_argument("--loading_device", type=str, default="cpu")
    parser.add_argument("--working_device", type=str, default="cpu")
    parser.add_argument("--save_to", type=str, default=None)
    parser.add_argument("--clip_l_save_to", type=str, default=None)
    parser.add_argument("--t5xxl_save_to", type=str, default=None)
    parser.add_argument("--models", type=str, nargs="*")
    parser.add_argument("--ratios", type=float, nargs="*")
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--diffusers", action="store_true") # dummy for compatibility
    return parser


def merge(args):
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []

    assert len(args.models) == len(args.ratios), "Number of models and ratios must be equal"

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    assert (
        args.save_to or args.clip_l_save_to or args.t5xxl_save_to
    ), "save_to or clip_l_save_to or t5xxl_save_to must be specified"
    dest_dir = os.path.dirname(args.save_to or args.clip_l_save_to or args.t5xxl_save_to)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    if args.flux_model is not None or args.clip_l is not None or args.t5xxl is not None:
        flux_state_dict, clip_l_state_dict, t5xxl_state_dict = merge_to_flux_model(
            args.loading_device,
            args.working_device,
            args.flux_model,
            args.clip_l,
            args.t5xxl,
            args.models,
            args.ratios,
            merge_dtype,
            save_dtype,
            args.mem_eff_load_save,
        )

        sai_metadata = None # metadata is simplified for standalone version

        if flux_state_dict is not None and len(flux_state_dict) > 0:
            save_to_file(args.save_to, flux_state_dict, save_dtype, sai_metadata, args.mem_eff_load_save)

        if clip_l_state_dict is not None and len(clip_l_state_dict) > 0:
            save_to_file(args.clip_l_save_to, clip_l_state_dict, save_dtype, None, args.mem_eff_load_save)

        if t5xxl_state_dict is not None and len(t5xxl_state_dict) > 0:
            save_to_file(args.t5xxl_save_to, t5xxl_state_dict, save_dtype, None, args.mem_eff_load_save)
    else:
        raise NotImplementedError("LoRA merge without base model is not implemented in standalone version")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO) # simplified logging setup
    merge(args)