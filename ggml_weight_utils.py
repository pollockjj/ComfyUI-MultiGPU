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
TENSORATOR_CACHE_SIZE_MB  = 12168
BUFFER_LOOK_AHEAD = 30

patch_cache = {}

cached_tensor_map = {}
level_one_tensors = [] 
level_two_tensors = []
ggml_tensor_buffers = []
dequantized_and_patched_tensor_buffers = []

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
    

    ggml_tensor_ptr = ggml_tensor.data_ptr()
    

    
    

    if ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level1" and cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor'] is not None:                # Immediately return if dequantized and patched tensor is already cached on compute_device
        return cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor']
    elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level2" and cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor'] is not None:              # Immediately copy.to() and return if dequantized and patched tensor is already cached on tensorator_device
        with torch.cuda.stream(tensorator_stream):
            level_two_tensor = cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor']
            level_two_tensor.to(compute_device, non_blocking=True)
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor
        
    elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "uninitialized":
        total_tensor_size = sum(info['tensor_size'] for info in cached_tensor_map.values())
        threshold = total_tensor_size * SMALL_TENSOR_THRESHOLD

        for tensor, info in cached_tensor_map.items():
            if info['cache_level'] == "uninitialized" and info['tensor_size'] < threshold:                            # If tensor is small, assign to level1
                info['cache_level'] = "level1"

        all_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                        if info['cache_level'] == "uninitialized"]


        all_tensors.sort(key=lambda x: (-x[1]['patch_qty'], x[1]['tensor_size']))                                     # Sort by patches descending, size ascending, the point is to maximize the total patches and then the total naked tensors represented in the allowed memory

        cumulative_size = 0
        for tensor, info in all_tensors:
            cumulative_size += info['tensor_size']
            if cumulative_size <= TENSORATOR_CACHE_SIZE_MB:
                cached_tensor_map[tensor]['cache_level'] = "level2"
            else:
                cached_tensor_map[tensor]['cache_level'] = "none"

        # Print the complete assignment table
        print("\n%-12s %-8s %-8s %-10s %-8s" % ("Pointer", "Index", "Patches", "Size(MB)", "Cache"))
        print("-" * 55)

        for ptr, info in sorted(cached_tensor_map.items(), key=lambda x: x[1]['index']):
            print("0x%-10x %-8d %-8d %-10.2f %-8s" %
                    (ptr, info['index'], info['patch_qty'], info['tensor_size'], info['cache_level']))


    #TODO: Kick off a background thread prefetch tensors to the tensorator_device = if ggml_tensor index + BUFFER_LOOK_AHEAD not in ggml_tensor_buffers and not in level_one_tensors or level_two_tensors kick off background prefetch of all tensors in that range
    # This way we have a copy local that is ready to go when we need it and we just look for it in that buffer and wait for it if it isn't there./

    with torch.cuda.stream(tensorator_stream):                                                                         # Start of uncached tensorator pipeline
        # tensorator_ggml = ggml_tensor.to(device=tensorator_device, non_blocking=True)
        
        patch_list = []
        for func, item, key in getattr(ggml_tensor, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches

        tensorator_tensor = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)
        if GGMLTensor is not None and isinstance(tensorator_tensor, GGMLTensor):
            tensorator_tensor.__class__ = torch.Tensor

        if patch_list:
            if patch_dtype is None:
                tensorator_tensor = func(patch_list, tensorator_tensor, key)
            else:
                tensorator_tensor = func(patch_list, tensorator_tensor, key, dtype if patch_dtype=="target" else patch_dtype)

        if ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level1":                #second time through for a level1-assigned tensor as level 1 branches after the first time
            level_one_tensor = tensorator_tensor.clone().to(compute_device, non_blocking=True)
            level_one_tensors.append(level_one_tensor)
            cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor'] = level_one_tensor
            # print(f"Moving GGML Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f} | to compute_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_one_tensor
        elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level2":
            level_two_tensor = tensorator_tensor.clone().to(tensorator_device, non_blocking=True)
            level_two_tensors.append(level_two_tensor)
            cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor'] = level_two_tensor
            # print(f"Moving GGML Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f} | to tensorator_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor

        tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
        tensorator_event.record(tensorator_stream)

        if ggml_tensor_ptr not in cached_tensor_map:
            cached_tensor_map[ggml_tensor_ptr] = {}
            cached_tensor_map[ggml_tensor_ptr]['index'] = len(cached_tensor_map) - 1
            cached_tensor_map[ggml_tensor_ptr]['patch_qty'] = len(patch_list)
            cached_tensor_map[ggml_tensor_ptr]['tensor_size'] = (tensorator_tensor.numel() * tensorator_tensor.element_size() / (1024 * 1024))
            cached_tensor_map[ggml_tensor_ptr]['cache_level'] = "uninitialized" # uninitialized, none, level1, level2
            cached_tensor_map[ggml_tensor_ptr]['dequantized_and_patched_tensor'] = None
            # print(f"GGML Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Patches: {cached_tensor_map[ggml_tensor_ptr]['patch_qty']:2d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f}")
    
    torch.cuda.current_stream().wait_event(tensorator_event)
    return tensorator_tensor