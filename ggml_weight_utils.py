import torch
import time
import weakref
import numpy as np
from collections import deque
import importlib
import sys
import torch.cuda.nvtx as nvtx 
import gguf

GGMLTensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops').GGMLTensor
dequantize_tensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant').dequantize_tensor


patch_cache = {}

cached_tensor_map = {}
cached_tensors = [] 
level_zero_tensors = []
prev_ggml_tensor_ptr = None

# hard-coded streams and variables for compute and tensorator during development
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:1")
compute_device = torch.device("cuda:0")
tensorator_device = torch.device("cuda:1")

# Setup events for cross-device synchronization
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)

def move_patch_to_tensorator(item):
    stream = tensorator_stream
    if isinstance(item, torch.Tensor):
        if stream is not None:
            with torch.cuda.stream(stream):
                return item.to(tensorator_device, non_blocking=True)
        else:
            return item.to(tensorator_device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_tensorator(x) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_tensorator(x) for x in item]
    else:
        return item

def retrieve_cached_patch(patches_item, key):
    cache_key = tuple(key) if isinstance(key, (list, tuple)) else key
    if cache_key in patch_cache:
        return patch_cache[cache_key]
    patch = move_patch_to_tensorator(patches_item)
    patch_cache[cache_key] = patch
    return patch

@profile
def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global prev_ggml_tensor_ptr
       
    nvtx.range_push("get_weight entry")
    
    # Check if tensor is None
    if tensor is None:
        nvtx.range_pop()
        return None
    
    ggml_tensor_ptr = tensor.data_ptr()

#    if ggml_tensor_ptr in cached_tensor_map:


    with torch.cuda.stream(tensorator_stream):
        nvtx.range_push("tensorator_ggml")
        tensorator_ggml = tensor.to(device=tensorator_device, non_blocking=True)
        nvtx.range_pop()
        
        nvtx.range_push("tensorator_retrieve_patches")
        patch_list = []
        for func, item, key in getattr(tensorator_ggml, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches
        nvtx.range_pop()
        
        nvtx.range_push("tensorator_tensor")
        tensorator_tensor = dequantize_tensor(tensorator_ggml, dtype, dequant_dtype)
        if GGMLTensor is not None and isinstance(tensorator_tensor, GGMLTensor):
            tensorator_tensor.__class__ = torch.Tensor
        nvtx.range_pop()
               
        nvtx.range_push("tensorator_tensor_apply_patches")
        if patch_list:
            if patch_dtype is None:
                tensorator_tensor = func(patch_list, tensorator_tensor, key)
            else:
                tensorator_tensor = func(patch_list, tensorator_tensor, key, dtype if patch_dtype=="target" else patch_dtype)
        nvtx.range_pop()
        
        nvtx.range_push("tensorator_transfer_to_compute")
        tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
        tensorator_event.record(tensorator_stream)
        nvtx.range_pop()
    
    nvtx.range_push("tensorator_wait_for_stream")
    torch.cuda.current_stream().wait_event(tensorator_event)
    nvtx.range_pop()

    if ggml_tensor_ptr not in cached_tensor_map:
        cached_tensor_map[ggml_tensor_ptr] = {}
        cached_tensor_map[ggml_tensor_ptr]['index'] = len(cached_tensor_map) - 1
        print(f"GGML Tensor {ggml_tensor_ptr}. GGML Map Index {cached_tensor_map[ggml_tensor_ptr]['index']}")


    return tensorator_tensor