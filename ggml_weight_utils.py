import torch
import time
import weakref
import numpy as np
from collections import deque
from .dequant import dequantize_tensor

try:
    from .ggml_tensor import GGMLTensor
except ImportError:
    GGMLTensor = None

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
        return
        
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
                    prefetch_buffers[inactive_buffer][tensor_ptr] = tensor.clone().to("cuda:0", non_blocking=True)
    
    # Clear the batch list for next time
    next_batch_to_prefetch[inactive_buffer] = []

@profile
def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global cached_tensor_map, cached_tensors, level_zero_tensors
    global active_buffer_index, prefetch_buffers, prefetch_batch_size, current_tensor_in_batch, next_batch_to_prefetch
    global total_tensors_processed, tensor_printed
    
    if tensor is None:
        return None
    
    ggml_tensor_ptr = tensor.data_ptr()

    # Check for Level 0 cache (permanent)
    if ggml_tensor_ptr in cached_tensor_map and 'level_zero_cache_location' in cached_tensor_map[ggml_tensor_ptr]:
        return cached_tensor_map[ggml_tensor_ptr]['level_zero_cache_location']
    
    # Check active prefetch buffer
    active_buffer = prefetch_buffers[active_buffer_index]
    if ggml_tensor_ptr in active_buffer:
        # Found in active buffer
        current_tensor_in_batch += 1
        
        # If we've used 2/3 of the active buffer, start filling the inactive one
        if current_tensor_in_batch == prefetch_batch_size // 2:
            # Start prefetching into inactive buffer
            prefetch_next_batch()
        
        # If we've used all tensors in active buffer, swap buffers
        if current_tensor_in_batch >= prefetch_batch_size:
            # Wait for prefetch to complete
            torch.cuda.synchronize("cuda:0")
            
            # Swap active buffer
            active_buffer_index = 1 - active_buffer_index
            current_tensor_in_batch = 0
        
        return active_buffer[ggml_tensor_ptr]
    
    # Not in prefetch, check Level 2 cache (cuda:1)
    if ggml_tensor_ptr in cached_tensor_map and 'level_two_cache_location' in cached_tensor_map[ggml_tensor_ptr]:
        with torch.cuda.stream(cuda1_stream):
            weight = cached_tensor_map[ggml_tensor_ptr]['level_two_cache_location']().clone()
            
        # Add to the appropriate next batch to prefetch
        inactive_buffer = 1 - active_buffer_index
        if len(next_batch_to_prefetch[inactive_buffer]) < prefetch_batch_size:
            next_batch_to_prefetch[inactive_buffer].append(ggml_tensor_ptr)
        
        return weight

    # Not in any cache, process normally
    patch_list = []
    device = tensor.device
    patches_data = getattr(tensor, "patches", [])
    for function, patches_item, key in patches_data:
        patch_result = retrieve_cached_patch(patches_item, device, key)
        patch_list += patch_result

    weight = dequantize_tensor(tensor, dtype, dequant_dtype)
    if GGMLTensor is not None and isinstance(weight, GGMLTensor):
        weight.__class__ = torch.Tensor
    
    # Print the first tensor before LoRA application
    if patch_list and not tensor_printed:
        # Save unpatched tensor
        clean_copy = weight.clone()
        
        # Print tensor shapes and a small sample of values
        print(f"\n\n===== First Tensor Analysis =====")
        print(f"Tensor Shape: {weight.shape}")
        
        # Get a small sample (5x5) of the tensor
        sample_rows = min(5, weight.shape[0])
        sample_cols = min(5, weight.shape[1])
        # Convert to float32 first to handle bfloat16
        pre_lora_sample = weight[:sample_rows, :sample_cols].detach().cpu().float().numpy()
        
        print("\nPre-LoRA values (5x5 sample):")
        for row in pre_lora_sample:
            print([f"{val:.6f}" for val in row])
        
        # Apply patches
        if patch_dtype is None:
            patched_weight = function(patch_list, weight, key)
        else:
            computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
            patched_weight = function(patch_list, weight, key, computed_patch_dtype)
        
        # Get post-LoRA sample
        post_lora_sample = patched_weight[:sample_rows, :sample_cols].detach().cpu().float().numpy()
        
        print("\nPost-LoRA values (5x5 sample):")
        for row in post_lora_sample:
            print([f"{val:.6f}" for val in row])
        
        # Get delta
        delta_sample = post_lora_sample - pre_lora_sample
        
        print("\nDelta values (5x5 sample):")
        for row in delta_sample:
            print([f"{val:.6f}" for val in row])
        
        # Calculate percentage of nonzeros
        delta_tensor = patched_weight - clean_copy
        nonzero_count = torch.count_nonzero(delta_tensor).item()
        nonzero_ratio = nonzero_count / delta_tensor.numel()
        print(f"\nNonzero elements: {nonzero_count} out of {delta_tensor.numel()} ({nonzero_ratio:.6f})")
        
        # Set the flag so we only print once
        tensor_printed = True
        
        # Continue with the existing logic
        weight = patched_weight
    elif patch_list:
        # Apply patches normally for other tensors
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
        with torch.cuda.stream(cuda0_stream):
            level0_tensor = weight.clone().to("cuda:0", non_blocking=True)
        
        level_zero_tensors.append(level0_tensor)
        
        if ggml_tensor_ptr not in cached_tensor_map:
            cached_tensor_map[ggml_tensor_ptr] = {}
        
        cached_tensor_map[ggml_tensor_ptr]['level_zero_cache_location'] = level0_tensor
    else:
        # Level 2 cache - on cuda:1
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