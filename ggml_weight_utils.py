import torch
import time
import weakref
import numpy as np
from collections import deque
import importlib
import sys
import torch.cuda.nvtx as nvtx 

cache_config = {'use_tensor_cache': False}

# Import from ComfyUI-GGUF
gguf_module = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant')
dequantize_tensor = gguf_module.dequantize_tensor
is_quantized = gguf_module.is_quantized

# Import GGMLTensor from ComfyUI-GGUF ops
try:
    ops_module = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops')
    GGMLTensor = ops_module.GGMLTensor
    move_patch_to_device_original = ops_module.move_patch_to_device
except (ImportError, AttributeError):
    GGMLTensor = None
    move_patch_to_device_original = None

patch_cache = {}

cached_tensor_map = {}
cached_tensors = [] 
level_zero_tensors = []
prev_ggml_tensor_ptr = None

# Ping-pong buffer for prefetching
active_buffer_index = 0  # 0 = buffer A, 1 = buffer B
prefetch_buffers = [{}, {}]  # Two dictionaries for ping-pong buffers
prefetch_batch_size = 15  # Size of each buffer
current_tensor_in_batch = 0  # Position in current batch
next_batch_to_prefetch = [[], []]  # Arrays to track which tensors to fetch next

use_level_zero_cache = False

# For tracking LoRA deltas
total_tensors_processed = 0
tensor_printed = False

# Create dedicated streams for each device
cuda0_stream = torch.cuda.Stream(device="cuda:0")
cuda1_stream = torch.cuda.Stream(device="cuda:1")

# Setup events for cross-device synchronization
cuda0_event = torch.cuda.Event(enable_timing=False)
cuda1_event = torch.cuda.Event(enable_timing=False)

def move_patch_to_device(item, device):
    if "cuda:0" in str(device):
        stream = cuda0_stream
    elif "cuda:1" in str(device):
        stream = cuda1_stream
    else:
        stream = None
    if isinstance(item, torch.Tensor):
        if stream is not None:
            with torch.cuda.stream(stream):
                return item.to(device, non_blocking=True)
        else:
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

def prefetch_next_batch():
    global active_buffer_index, prefetch_buffers, next_batch_to_prefetch
    inactive_buffer = 1 - active_buffer_index
    if not next_batch_to_prefetch[inactive_buffer]:
        return
    prefetch_buffers[inactive_buffer].clear()
    with torch.cuda.stream(cuda0_stream):
        for tensor_ptr in next_batch_to_prefetch[inactive_buffer]:
            if tensor_ptr in cached_tensor_map and 'level_two_cache_location' in cached_tensor_map[tensor_ptr]:
                tensor = cached_tensor_map[tensor_ptr]['level_two_cache_location']()
                if tensor is not None:
                    prefetch_buffers[inactive_buffer][tensor_ptr] = tensor.clone().to("cuda:0", non_blocking=True)
    next_batch_to_prefetch[inactive_buffer] = []
@profile
def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
    nvtx.range_push("get_weight entry")
    if not cache_config["use_tensor_cache"]:
        nvtx.range_push("patch-transfer branch")
        if tensor is None:
            nvtx.range_pop()  # end patch-transfer branch
            nvtx.range_pop()  # end get_weight entry
            return None
        patch_list = []
        d = tensor.device
        for func, item, key in getattr(tensor, "patches", []):
            patch_list += retrieve_cached_patch(item, d, key)
        w = dequantize_tensor(tensor, dtype, dequant_dtype)
        if GGMLTensor is not None and isinstance(w, GGMLTensor):
            w.__class__ = torch.Tensor
        if patch_list:
            w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
            total_tensors_processed += 1
        nvtx.range_pop()  # end patch-transfer branch
        nvtx.range_pop()  # end get_weight entry
        return w

    # Full ping-pong caching branch (dynamically evaluated each call)
    if not hasattr(get_weight, "_first_call_logged"):
        print("\n" + "="*80)
        print("MultiGPU: Full tensor caching is ENABLED")
        print(f"Using prefetch batch size of {prefetch_batch_size}")
        print("="*80+"\n")
        get_weight._first_call_logged = True
    if tensor is None:
        nvtx.range_pop()
        return None
    ptr = tensor.data_ptr()
    if ptr in cached_tensor_map and "level_zero_cache_location" in cached_tensor_map[ptr]:
        nvtx.range_pop(); nvtx.range_pop()
        return cached_tensor_map[ptr]["level_zero_cache_location"]
    buf = prefetch_buffers[active_buffer_index]
    if ptr in buf:
        current_tensor_in_batch += 1
        if current_tensor_in_batch == prefetch_batch_size // 2:
            prefetch_next_batch()
        if current_tensor_in_batch >= prefetch_batch_size:
            torch.cuda.synchronize("cuda:0")
            active_buffer_index = 1 - active_buffer_index
            current_tensor_in_batch = 0
        nvtx.range_pop(); nvtx.range_pop()
        return buf[ptr]
    if ptr in cached_tensor_map and "level_two_cache_location" in cached_tensor_map[ptr]:
        with torch.cuda.stream(cuda1_stream):
            w = cached_tensor_map[ptr]["level_two_cache_location"]().clone()
        ib = 1 - active_buffer_index
        if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
            next_batch_to_prefetch[ib].append(ptr)
        nvtx.range_pop(); nvtx.range_pop()
        return w
    patch_list = []
    d = tensor.device
    for func, item, key in getattr(tensor, "patches", []):
        patch_list += retrieve_cached_patch(item, d, key)
    w = dequantize_tensor(tensor, dtype, dequant_dtype)
    if GGMLTensor is not None and isinstance(w, GGMLTensor):
        w.__class__ = torch.Tensor
    if patch_list:
        w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
        total_tensors_processed += 1
    if ptr % 5 == 0 and use_level_zero_cache:
        with torch.cuda.stream(cuda0_stream):
            l0 = w.clone().to("cuda:0", non_blocking=True)
        level_zero_tensors.append(l0)
        if ptr not in cached_tensor_map:
            cached_tensor_map[ptr] = {}
        cached_tensor_map[ptr]["level_zero_cache_location"] = l0
    else:
        with torch.cuda.stream(cuda1_stream):
            l2 = w.clone().to("cuda:1", non_blocking=True)
        cached_tensors.append(l2)
        if ptr not in cached_tensor_map:
            cached_tensor_map[ptr] = {}
        cached_tensor_map[ptr]["level_two_cache_location"] = weakref.ref(l2)
        ib = 1 - active_buffer_index
        if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
            next_batch_to_prefetch[ib].append(ptr)
    nvtx.range_pop()  # end full caching branch
    nvtx.range_pop()  # end get_weight entry
    return w
