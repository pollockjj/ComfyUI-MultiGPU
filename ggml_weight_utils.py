import torch
import time
import weakref
import numpy as np
from collections import deque
import importlib
import sys

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

cuda0_stream = torch.cuda.Stream(device="cuda:0")
cuda1_stream = torch.cuda.Stream(device="cuda:1")

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
    """Prefetch the next batch of tensors into the inactive buffer"""
    global active_buffer_index, prefetch_buffers, next_batch_to_prefetch
    
    # Determine which buffer to fill
    inactive_buffer = 1 - active_buffer_index
    
    # Skip if nothing to prefetch
    if not next_batch_to_prefetch[inactive_buffer]:
        print(f"[MultiGPU] No tensors to prefetch for buffer {inactive_buffer}")
        return
    
    print(f"[MultiGPU] Prefetching {len(next_batch_to_prefetch[inactive_buffer])} tensors into buffer {inactive_buffer}")
        
    # Clear the inactive buffer
    prefetch_buffers[inactive_buffer].clear()
    
    # Fill the inactive buffer
    with torch.cuda.stream(cuda0_stream):
        for tensor_ptr in next_batch_to_prefetch[inactive_buffer]:
            if tensor_ptr in cached_tensor_map and 'level_two_cache_location' in cached_tensor_map[tensor_ptr]:
                # Get tensor from cuda:1
                tensor = cached_tensor_map[tensor_ptr]['level_two_cache_location']()
                if tensor is not None:
                    # Transfer to cuda:0
                    print(f"[MultiGPU] Prefetching tensor {tensor_ptr} from L2 cache to buffer {inactive_buffer}")
                    prefetch_buffers[inactive_buffer][tensor_ptr] = tensor.clone().to("cuda:0", non_blocking=True)
                else:
                    print(f"[MultiGPU] WARNING: Tensor {tensor_ptr} was garbage collected from L2 cache")
    
    print(f"[MultiGPU] Prefetched {len(prefetch_buffers[inactive_buffer])} tensors into buffer {inactive_buffer}")
    
    # Clear the batch list for next time
    next_batch_to_prefetch[inactive_buffer] = []

def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global cached_tensor_map, cached_tensors, level_zero_tensors
    global active_buffer_index, prefetch_buffers, prefetch_batch_size, current_tensor_in_batch, next_batch_to_prefetch
    global total_tensors_processed, tensor_printed
    
    if tensor is None:
        return None
    
    # Debug message to confirm our function is being called instead of original
    print(f"[MultiGPU] Using enhanced get_weight with ping-pong buffer for tensor at {tensor.data_ptr()}")
    
    ggml_tensor_ptr = tensor.data_ptr()

    # Check for Level 0 cache (permanent)
    if ggml_tensor_ptr in cached_tensor_map and 'level_zero_cache_location' in cached_tensor_map[ggml_tensor_ptr]:
        print(f"[MultiGPU] L0 Cache HIT for tensor {ggml_tensor_ptr}")
        return cached_tensor_map[ggml_tensor_ptr]['level_zero_cache_location']
    
    # Check active prefetch buffer
    active_buffer = prefetch_buffers[active_buffer_index]
    if ggml_tensor_ptr in active_buffer:
        # Found in active buffer
        print(f"[MultiGPU] Prefetch buffer HIT for tensor {ggml_tensor_ptr} in buffer {active_buffer_index}")
        current_tensor_in_batch += 1
        
        # If we've used 2/3 of the active buffer, start filling the inactive one
        if current_tensor_in_batch == prefetch_batch_size // 2:
            # Start prefetching into inactive buffer
            print(f"[MultiGPU] Triggering prefetch of next batch, current usage: {current_tensor_in_batch}/{prefetch_batch_size}")
            prefetch_next_batch()
        
        # If we've used all tensors in active buffer, swap buffers
        if current_tensor_in_batch >= prefetch_batch_size:
            # Wait for prefetch to complete
            print(f"[MultiGPU] Swapping buffers: {active_buffer_index} -> {1-active_buffer_index}")
            torch.cuda.synchronize("cuda:0")
            
            # Swap active buffer
            active_buffer_index = 1 - active_buffer_index
            current_tensor_in_batch = 0
        
        return active_buffer[ggml_tensor_ptr]
    
    # Not in prefetch, check Level 2 cache (cuda:1)
    if ggml_tensor_ptr in cached_tensor_map and 'level_two_cache_location' in cached_tensor_map[ggml_tensor_ptr]:
        print(f"[MultiGPU] L2 Cache HIT for tensor {ggml_tensor_ptr}")
        with torch.cuda.stream(cuda1_stream):
            weight = cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location']().clone()
            
        # Add to the appropriate next batch to prefetch
        inactive_buffer = 1 - active_buffer_index
        if len(next_batch_to_prefetch[inactive_buffer]) < prefetch_batch_size:
            next_batch_to_prefetch[inactive_buffer].append(ggml_tensor_ptr)
        
        return weight

    # Not in any cache, process normally
    print(f"[MultiGPU] CACHE MISS for tensor {ggml_tensor_ptr}")
    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        patch_result = retrieve_cached_patch(patches_item, device, key)
        patch_list += patch_result

    weight = dequantize_tensor(tensor, dtype, dequant_dtype)
    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    
    # Process patches if any
    if patch_list:
        # Apply patches
        if patch_dtype is None:
            weight = function(patch_list, weight, key)
        else:
            computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
            weight = function(patch_list, weight, key, computed_patch_dtype)
        
        # Count tensors processed
        total_tensors_processed += 1

    # Add to appropriate cache
    if ggml_tensor_ptr % 5 == 0 and use_level_zero_cache:
        # Level 0 cache - direct on cuda:0
        print(f"[MultiGPU] Adding tensor {ggml_tensor_ptr} to L0 cache")
        with torch.cuda.stream(cuda0_stream):
            level0_tensor = weight.clone().to("cuda:0", non_blocking=True)
        
        level_zero_tensors.append(level0_tensor)
        
        if ggml_tensor_ptr not in cached_tensor_map:
            cached_tensor_map[ggml_tensor_ptr] = {}
        
        cached_tensor_map[ggml_tensor_ptr]['level_zero_cache_location'] = level0_tensor
    else:
        # Level 2 cache - on cuda:1
        print(f"[MultiGPU] Adding tensor {ggml_tensor_ptr} to L2 cache and prefetch queue")
        with torch.cuda.stream(cuda1_stream):
            level2_tensor = weight.clone().to("cuda:1", non_blocking=True)
        
        cached_tensors.append(level2_tensor)
        
        if ggml_tensor_ptr not in cached_tensor_map:
            cached_tensor_map[ggml_tensor_ptr] = {}
        
        cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location'] = weakref.ref(level2_tensor)
        
        # Add to next batch to prefetch
        inactive_buffer = 1 - active_buffer_index
        if len(next_batch_to_prefetch[inactive_buffer]) < prefetch_batch_size:
            next_batch_to_prefetch[inactive_buffer].append(ggml_tensor_ptr)
    
    return weight