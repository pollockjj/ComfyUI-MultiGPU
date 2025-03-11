import torch
import time
import importlib
import comfy.model_management

dequantize_tensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant').dequantize_tensor

try:
    GGMLTensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.ggml_tensor').GGMLTensor
except ImportError:
    GGMLTensor = None

# Global variable for tracking inference order
cast_bias_weight_inf_ord = 0

patch_cache = {}

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item

def retrieve_cached_patch(patches_item, device, key):
    cache_key = tuple(key) if isinstance(key, (list, tuple)) else key
    if cache_key in patch_cache:
        return patch_cache[cache_key]
    patch = move_patch_to_device(patches_item, device)
    patch_cache[cache_key] = patch
    return patch

def compute_size(item):
    if isinstance(item, torch.Tensor):
        return item.numel() * item.element_size()
    elif isinstance(item, (list, tuple)):
        return sum(compute_size(x) for x in item)
    else:
        return 0

def cast_bias_weight_patched(s, input=None, dtype=None, device=None, bias_dtype=None):
    global cast_bias_weight_inf_ord
    from . import cached_tensor_map
    
    if input is not None:
        if dtype is None:
            dtype = getattr(input, "dtype", torch.float32)
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    # Get the original hash for lookup and tracking
    stored_hash = s.weight.original_hash
    
    # Track inference order if this is the first time through
    if stored_hash in cached_tensor_map and cached_tensor_map[stored_hash]['cache_level'] == "pre-inference":
        cached_tensor_map[stored_hash]['inf_order'] = cast_bias_weight_inf_ord
        cast_bias_weight_inf_ord += 1
        cached_tensor_map[stored_hash]['cache_level'] = "uninitialized"
        print(f"TENSOR: ptr=0x{stored_hash:x} | index={cached_tensor_map[stored_hash]['inf_order']:<4} | name={cached_tensor_map[stored_hash]['name']:<60} | device={cached_tensor_map[stored_hash]['distorch_device']:<8} | size={cached_tensor_map[stored_hash]['tensor_size']:>8.2f}")

    # Standard processing
    weight_to = s.weight.to(device)

    bias = None
    non_blocking = comfy.model_management.device_supports_non_blocking(device)
    if s.bias is not None:
        bias = s.get_weight(s.bias.to(device), dtype)
        bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

    weight = s.get_weight(weight_to, dtype)
    weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    
    return weight, bias

def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    if tensor is None:
        return None
    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        patch_result = retrieve_cached_patch(patches_item, device, key)
        patch_list += patch_result
    weight = dequantize_tensor(tensor, dtype, dequant_dtype)
    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    if patch_list:
        if patch_dtype is None:
            weight = function(patch_list, weight, key)
        else:
            computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
            weight = function(patch_list, weight, key, computed_patch_dtype)
    return weight