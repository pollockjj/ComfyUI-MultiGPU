import torch
import time
import weakref
import numpy as np
from collections import deque
import importlib
import sys
import torch.cuda.nvtx as nvtx 
import gguf
import gc

GGMLTensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops').GGMLTensor
dequantize_tensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant').dequantize_tensor

SMALL_TENSOR_THRESHOLD = 0.0001  # 0.01% of total size
TENSORATOR_CACHE_SIZE_MB  = 12168
TENSORATOR_GGML_CACHE_SIZE_MB  = 4096

patch_cache = {}

cached_tensor_map = {}
level_one_tensors = [] 
level_two_tensors = []
level_three_tensors = []
cached_tensor_buffers = []
prefetch_candidate_stack = []

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

def initialize_cache_levels():
    global prefetch_candidate_stack
    total_tensor_size = sum(info['tensor_size'] for info in cached_tensor_map.values())
    threshold = total_tensor_size * SMALL_TENSOR_THRESHOLD

    for tensor, info in cached_tensor_map.items():
        if info['cache_level'] == "uninitialized" and info['tensor_size'] < threshold:
            info['cache_level'] = "level1"

    all_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                    if info['cache_level'] == "uninitialized"]

    all_tensors.sort(key=lambda x: (-x[1]['patch_qty'], x[1]['tensor_size']))

    cumulative_size = 0
    level2_size = 0
    level3_size = 0
    for tensor, info in all_tensors:
        cumulative_size += info['tensor_size']
        if level2_size <= TENSORATOR_CACHE_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "level2"
            level2_size += info['tensor_size']
        elif level3_size <= TENSORATOR_GGML_CACHE_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "level3"
            #print(f"Assigning GGML Tensor 0x{tensor:x} to level3 cache | Size: {info['tensor_size']:.2f}MB")
            level3_size += info['tensor_size']
        else:
            cached_tensor_map[tensor]['cache_level'] = "none"

@profile
def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None):

    if ggml_tensor is None:                                                                                           # Check if tensor is None
        return None

    ggml_tensor_ptr = ggml_tensor.data_ptr()

    print(f"Tensor 0x{ggml_tensor_ptr:x} | RefCount: {sys.getrefcount(ggml_tensor)-1} | Device: {ggml_tensor.device} | Referrers: {len(gc.get_referrers(ggml_tensor))}")

    if ggml_tensor_ptr in cached_tensor_map:
        if cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level1" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is not None:                # Immediately return if dequantized and patched tensor is already cached on compute_device
            return cached_tensor_map[ggml_tensor_ptr]['cached_tensor']
        elif cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level2" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is not None:              # Immediately copy.to() and return if dequantized and patched tensor is already cached on tensorator_device
            with torch.cuda.stream(tensorator_stream):
                level_two_tensor = cached_tensor_map[ggml_tensor_ptr]['cached_tensor']
                level_two_tensor.to(compute_device, non_blocking=True)
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
                return level_two_tensor
        elif cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "uninitialized":
            initialize_cache_levels()

    with torch.cuda.stream(tensorator_stream):                                                                         # Start of uncached tensorator pipeline

        patch_list = []
        for func, item, key in getattr(ggml_tensor, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches
            
        if ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level3" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is not None:
            tensorator_tensor = dequantize_tensor(cached_tensor_map[ggml_tensor_ptr]['cached_tensor'], dtype, dequant_dtype)
        else:
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
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = level_one_tensor
            # print(f"Moving Dequantized and Patched Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f} | to compute_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_one_tensor
        elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level2":
            level_two_tensor = tensorator_tensor.clone().to(tensorator_device, non_blocking=True)
            level_two_tensors.append(level_two_tensor)
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = level_two_tensor
            # print(f"Moving Dequantized and Patched Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f} | to tensorator_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor
        elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level3" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is None:
            level_three_tensor = ggml_tensor.to(tensorator_device, non_blocking=True)
            level_three_tensors.append(level_three_tensor)
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = level_three_tensor
            #print(f"Moving GGML Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f} | to tensorator_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)

        tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
        tensorator_event.record(tensorator_stream)
        

        if ggml_tensor_ptr not in cached_tensor_map:
            cached_tensor_map[ggml_tensor_ptr] = {}
            cached_tensor_map[ggml_tensor_ptr]['index'] = len(cached_tensor_map) - 1
            cached_tensor_map[ggml_tensor_ptr]['patch_qty'] = len(patch_list)
            cached_tensor_map[ggml_tensor_ptr]['tensor_size'] = (tensorator_tensor.numel() * tensorator_tensor.element_size() / (1024 * 1024))
            cached_tensor_map[ggml_tensor_ptr]['cache_level'] = "uninitialized" # uninitialized, none, level1, level2, number 0...BUFFER_LOOK_AHEAD - 1
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = None
            # print(f"GGML Tensor: 0x{ggml_tensor_ptr:x} | Index: {cached_tensor_map[ggml_tensor_ptr]['index']:3d} | Patches: {cached_tensor_map[ggml_tensor_ptr]['patch_qty']:2d} | Size: {cached_tensor_map[ggml_tensor_ptr]['tensor_size']:.2f}")
    
    torch.cuda.current_stream().wait_event(tensorator_event)
    return tensorator_tensor