import torch
import time
import weakref
import numpy as np
from collections import deque
import importlib
import sys
import torch.cuda.nvtx as nvtx 

# Configuration options
cache_config = {
    'use_tensor_cache': False,
}

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
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:1")

# Setup events for cross-device synchronization
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)

def move_patch_to_device(item, device):
    if "cuda:0" in str(device):
        stream = compute_stream
    elif "cuda:1" in str(device):
        stream = tensorator_stream
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

@profile
def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
    global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
       
    nvtx.range_push("get_weight entry")
    
    # Check if tensor is None
    if tensor is None:
        nvtx.range_pop()
        return None
    
    # Phase 1: Basic Linear Pipeline Implementation. In an ideal world, the entirety of `compute`'s VRAM is available for compute over latent space. Current HunyuanVideo and Wan Video allow for the entirety of a card's space to be used. Assume that is the case. That there is a buffer of VRAM in `compute` that will be used for the active tensor and a buffer of N+1, N+2 tensors ready to go. Current ping-pong buffer implemnented below is 15 tensors per.
    nvtx.range_push("tensorator_processing")
    
    # Keep GGML layer on loaded (compute or other if DisTorch) device. The gguf library code appears to funnel GGML layers through compute device, so until those are copied or cached it will be the fastest way. 
    # #TODO: Build a GGML-Layer buffered cache. The best place to store GGML (dead) Layers is the slowest device you can tolerate = DRAM if you buffer it properly onto tensorator 
    tensorator_device = torch.device("cuda:1")
    with torch.cuda.stream(tensorator_stream):
        
        tensorator_tensor = tensor.to(device=tensorator_device, non_blocking=True)
    
        # Step 2: Prepare and cache patches on tensorator
        patch_list = []
        for func, item, key in getattr(tensorator_tensor, "patches", []):
            patches = retrieve_cached_patch(item, tensorator_device, key)
            patch_list += patches
        
        # Step 3: Dequantize tensor on tensorator, using temporary tensor that currently gets garbage collected soon afterwards. Curent flow has this repeated every inference step, every tensor
        w = dequantize_tensor(tensorator_tensor, dtype, dequant_dtype)
        if GGMLTensor is not None and isinstance(w, GGMLTensor):
            w.__class__ = torch.Tensor
        
        # Step 4: Apply patches on tensorator - from everything I have read, everything I have tried, everything I have seen implemented elsewhere like Forge, this is the only way to apply LoRAs to a tensor in an efficient memory-saving manner. It must happen every dequantization
        #TODO: Implement a version of the caching routines that explicitly uses tensorator to its fullest extent. This means VRAM filled with cached, fully-patched tensors ready to be used without any additional compute while the GPU time is at 100% on keeping all of the tensorator pipeline buffers filled. Only time compute on `tensorator` should be idle is exact that: All buffers full.
        if patch_list:
            if patch_dtype is None:
                w = func(patch_list, w, key)
            else:
                w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
            total_tensors_processed += 1
        
        # Step 5: Transfer result back to compute
        #TODO: DMA might be the better way to go. It appears to be the fastest way for us to operate on the tensor sent to get_weight. We now have our own tensor (w) that is on tensorator. In this linear pipeline, is it actually faster to not do this step and just have `compute` pull it from tensorator via DMA? I suspect it is, especially with my NVLink. Could be wrong but the results from step one were suprising (thus its elimination) and need for further investigation.
        w = w.to(device="cuda:0", non_blocking=True)
        
        # Record completion event
        tensorator_event.record(tensorator_stream)
    
    # Wait for tensorator to finish
    torch.cuda.current_stream().wait_event(tensorator_event)
    
    nvtx.range_pop()  # end tensorator_processing
    nvtx.range_pop()  # end get_weight entry
    
    # Return the fully processed weight
    return w
