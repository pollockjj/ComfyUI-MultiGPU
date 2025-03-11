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
cached_tensor_map = {}

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
    from . import distorch_load_map
    
    if input is not None:
        if dtype is None:
            dtype = getattr(input, "dtype", torch.float32)
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    ggml_tensor_hash = s.weight.original_hash

    if ggml_tensor_hash in distorch_load_map and distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] is False:
        distorch_load_map[ggml_tensor_hash]['inf_order'] = cast_bias_weight_inf_ord
        cast_bias_weight_inf_ord += 1
        distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] = True
  
  
    # if cached_tensor_map[ggml_tensor_hash]['cache_level'] = "ggml_prefetch_buffer" we should skip.
    weight_to = s.weight.to(device)

    bias = None
    non_blocking = comfy.model_management.device_supports_non_blocking(device)
    if s.bias is not None:
        bias = s.get_weight(s.bias.to(device), dtype)
        bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

    kwargs = {}
    if ggml_tensor_hash in distorch_load_map and 'inf_order' in distorch_load_map[ggml_tensor_hash]:
        kwargs['index'] = distorch_load_map[ggml_tensor_hash]['inf_order']
    if ggml_tensor_hash in distorch_load_map:
        kwargs['name'] = distorch_load_map[ggml_tensor_hash]['name']
    if ggml_tensor_hash in distorch_load_map and 'distorch_device' in distorch_load_map[ggml_tensor_hash]:
        kwargs['distorch_device'] = distorch_load_map[ggml_tensor_hash]['distorch_device']
    kwargs['ggml_tensor_hash'] = ggml_tensor_hash
    
    try:
        weight = s.get_weight(weight_to, dtype, **kwargs)
    except TypeError:
        weight = s.get_weight(weight_to, dtype)
    weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    
    return weight, bias

def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None, index=None, name=None, ggml_tensor_hash=None, distorch_device=None):
    global cached_tensor_map
    if ggml_tensor is None:
        return None

    patch_list = []
    device = ggml_tensor.device
    patches_data = getattr(ggml_tensor, "patches", [])
    for function, patches_item, key in patches_data:
        patch_result = retrieve_cached_patch(patches_item, device, key)
        patch_list += patch_result

    
    # If cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized" then I want to set up the GGML buffer. This deterministic. I need the tensor on the CPU that is N positions ahead of the current index and we need to
    # prefetch it to a "ggml_prefetch_buffer" collection of hardrefs until we use it. So:
    # 1. Filter out all tensor that are not on CPU
    # 2. Sort the tensors by index
    # 3. Find the tensor that is N positions ahead of the current index
    # 4. Prefetch it to the "ggml_prefetch_buffer" collection of hardrefs, non-blocking
    # 5. Set the cache_level to "ggml_prefetch"
    # 6. Look to see if my own tensor is in the "ggml_prefetch_buffer" collection of hardrefs (which it will be after the first N tensors are prefetched)
    # 7. If it is swap out this reference for the one on remote, slow DRAM
    # 8. If it is not, do nothing. The normal flow will fetch the tensor (slowly, from DRAM) and process it as normal.

    weight = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)

    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    if patch_list:
        if patch_dtype is None:
            weight = function(patch_list, weight, key)
        else:
            computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
            weight = function(patch_list, weight, key, computed_patch_dtype)
            
    if ggml_tensor_hash not in cached_tensor_map and ggml_tensor_hash is not None:
        cached_tensor_map[ggml_tensor_hash] = {}
        cached_tensor_map[ggml_tensor_hash]['index'] = index
        cached_tensor_map[ggml_tensor_hash]['name'] = name
        cached_tensor_map[ggml_tensor_hash]['patch_qty'] = len(patch_list)
        cached_tensor_map[ggml_tensor_hash]['tensor_size'] = (ggml_tensor.numel() * ggml_tensor.element_size() / (1024 * 1024))
        cached_tensor_map[ggml_tensor_hash]['distorch_device'] = distorch_device
        cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized"
        cached_tensor_map[ggml_tensor_hash]['cached_tensor'] = None
        print(f"TENSOR CACHE: ptr=0x{ggml_tensor_hash:x} | index={index} | name={name} | patches={len(patch_list)} | device={distorch_device} | size={cached_tensor_map[ggml_tensor_hash]['tensor_size']:.2f}MB")
  
    return weight