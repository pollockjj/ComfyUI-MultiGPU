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

SMALL_TENSOR_THRESHOLD = 0.0001  # 0.01% of total size
TENSORATOR_CACHE_SIZE_MB  = 2048

patch_cache = {}

cached_tensor_map = {}
cached_tensors = [] 
level_zero_tensors = []

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
def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None):

    if ggml_tensor is None:                                                                                           # Check if tensor is None
        return None

    if ggml_tensor in cached_tensor_map and cached_tensor_map[ggml_tensor]['cache_level'] == "level1":                # Immediately return if dequantized and patched tensor is already cached on compute_device
        return cached_tensor_map[ggml_tensor]['dequantized_and_patched_tensor']
    elif ggml_tensor in cached_tensor_map and cached_tensor_map[ggml_tensor]['cache_level'] == "level2":              # Immediately copy.to() and return if dequantized and patched tensor is already cached on tensorator_device
        with torch.cuda.stream(tensorator_stream):
            level_two_tensor = cached_tensor_map[ggml_tensor]['dequantized_and_patched_tensor']
            level_two_tensor.to(compute_device, non_blocking=True)
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor
    elif ggml_tensor in cached_tensor_map and cached_tensor_map[ggml_tensor]['cache_level'] == "unmapped":
        total_tensor_size = sum(info['tensor_size'] for info in cached_tensor_map.values())
        if cached_tensor_map[ggml_tensor]['tensor_size'] <  (total_tensor_size * SMALL_TENSOR_THRESHOLD):              # Level 1 cache - store tiny tensors directly on compute_device with hard reference
            cached_tensor_map[ggml_tensor]['cache_level'] = "level1"
    else:  # Level 2 cache - store on tensorator_device with future prefetching to level 1
        all_tensors = [(t, info) for t, info in cached_tensor_map.items() if info['cache_level'] == "unmapped"]
        all_tensors.sort(key=lambda x: (-x[1]['patch_qty'], x[1]['tensor_size']))
        cumulative_size = 0
        for i, (t, info) in enumerate(all_tensors):
            cumulative_size += info['tensor_size']
            info['patch_priority'] = i
            if cumulative_size <= TENSORATOR_CACHE_SIZE_MB:
                info['cache_level'] = "level2"
            else:
                info['cache_level'] = "none"

        # Verify that the original dictionary was modified
        print(f"DEBUG: Current tensor {ggml_tensor.data_ptr():x} cache_level is now: {cached_tensor_map[ggml_tensor]['cache_level']}")
        # Check a few other tensors from all_tensors to verify they were updated too
        if len(all_tensors) > 0:
            sample_tensor, sample_info = all_tensors[0]
            print(f"DEBUG: Sample tensor {sample_tensor.data_ptr():x} cache_level is now: {cached_tensor_map[sample_tensor]['cache_level']}")


    with torch.cuda.stream(tensorator_stream):                                                                         # Start of uncached tensorator pipeline
        tensorator_ggml = ggml_tensor.to(device=tensorator_device, non_blocking=True)
        
        patch_list = []
        for func, item, key in getattr(tensorator_ggml, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches

        tensorator_tensor = dequantize_tensor(tensorator_ggml, dtype, dequant_dtype)
        if GGMLTensor is not None and isinstance(tensorator_tensor, GGMLTensor):
            tensorator_tensor.__class__ = torch.Tensor

        if patch_list:
            if patch_dtype is None:
                tensorator_tensor = func(patch_list, tensorator_tensor, key)
            else:
                tensorator_tensor = func(patch_list, tensorator_tensor, key, dtype if patch_dtype=="target" else patch_dtype)

        if ggml_tensor in cached_tensor_map and cached_tensor_map[ggml_tensor]['cache_level'] == "level1":
            cached_tensor_map[ggml_tensor]['dequantized_and_patched_tensor'] = tensorator_tensor.clone().to(compute_device, non_blocking=True)
            print(f"Moving GGML Tensor: {ggml_tensor.data_ptr():x} | Index: {cached_tensor_map[ggml_tensor]['index']:3d} | Size: {cached_tensor_map[ggml_tensor]['tensor_size']:.2f} | to compute_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return cached_tensor_map[ggml_tensor]['dequantized_and_patched_tensor']
        elif ggml_tensor in cached_tensor_map and cached_tensor_map[ggml_tensor]['cache_level'] == "level2":
            with torch.cuda.stream(tensorator_stream):
                level_two_tensor = cached_tensor_map[ggml_tensor]['dequantized_and_patched_tensor'].to(compute_device, non_blocking=True)
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
                return level_two_tensor


        tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
        tensorator_event.record(tensorator_stream)

        if ggml_tensor not in cached_tensor_map:
            cached_tensor_map[ggml_tensor] = {}
            cached_tensor_map[ggml_tensor]['index'] = len(cached_tensor_map) - 1
            cached_tensor_map[ggml_tensor]['patch_qty'] = len(patch_list)
            cached_tensor_map[ggml_tensor]['tensor_size'] = (tensorator_tensor.numel() * tensorator_tensor.element_size() / (1024 * 1024))
            cached_tensor_map[ggml_tensor]['cache_level'] = "unmapped" # unmapped, none, level1, level2
            cached_tensor_map[ggml_tensor]['dequantized_and_patched_tensor'] = None
            print(f"GGML Tensor: {ggml_tensor.data_ptr():x} | Index: {cached_tensor_map[ggml_tensor]['index']:3d} | Patches: {cached_tensor_map[ggml_tensor]['patch_qty']:2d} | Size: {cached_tensor_map[ggml_tensor]['tensor_size']:.2f}")
    
    torch.cuda.current_stream().wait_event(tensorator_event)
    return tensorator_tensor