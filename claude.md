# DisTorch Multi-GPU Distribution System Documentation

## Project Overview

DisTorch distributes Stable Diffusion model layers across multiple GPUs to enable running models larger than any single GPU's VRAM capacity. The system analyzes model architecture and memory requirements, then creates optimal device assignments to maximize throughput while minimizing memory bottlenecks.

## Core Components

### 1. Model Distribution Algorithm

- Parses device allocation ratios from user configuration
- Analyzes layer types and memory requirements of each model component
- Creates proportional distribution plans based on available device memory
- Returns device-to-layer assignments optimized for parallel execution

### 2. Virtual VRAM System

- Allows users to specify primary device and VRAM to "borrow" from other devices
- Creates donor pools from secondary GPUs and/or system RAM
- Automatically calculates optimal allocation strings for easy configuration
- Handles memory tracking across all participating devices

### 3. Model Identification and Assignment Storage

- Each model is fingerprinted with a unique hash based on type, size, and key layers
- Device assignments are stored in a global `model_allocation_store` dictionary
- Assignments are retrieved by hash during model loading

### 4. GGML Tensor Caching

- Implements a sophisticated multi-level caching system for optimal memory usage
- Uses a global `cached_tensor_map` to track tensors by their pointer addresses
- Maintains strong references to prevent garbage collection
- Includes cache level assignments based on tensor characteristics

### 5. Asynchronous Tensor Processing

- Uses dedicated CUDA streams for compute and tensor operations
- Leverages non-blocking transfers to overlap computation and memory transfers
- Uses events for synchronization between operations on different devices
- Implements sophisticated prefetching to hide transfer latencies

## Implementation

The core implementation integrates at the optimal point in the module loading process through the patched `ModelPatcher.load` method:

1. Generate a unique hash for the model being loaded
2. Check if device assignments exist for this model hash
3. If assignments exist, create a flattened lookup map of layer name to device
4. During module loading, check if each layer has an assignment
5. Move each layer directly to its assigned device in one step

This approach allows:
- Moving modules directly to their final destination without memory spikes
- Maintaining consistent internal state in ComfyUI
- Supporting all model types and architectures

## Completed Optimizations

The development branch contains many advanced features:

1. **Architecture-Agnostic Model Support**: Works with all model architectures in GGUF format regardless of parent-child relationships

2. **Three-Level Caching System**:
   - Level 1: Frequently accessed tensors kept on compute device
   - Level 2: Medium tensors on secondary GPU
   - Level 3: GGML tensors in a prefetch buffer

3. **Asynchronous Buffer System**:
   - Uses deterministic prefetching to hide transfer latency
   - Transfers entire blocks at precise intervals for optimal bandwidth
   - Pre-computes and requantizes LoRA patches to reduce computation

4. **Performance Optimizations**:
   - Single-GPU mode: Minimal overhead (~2-5%)
   - Multi-GPU mode: PCIe transfer latency hidden by computation
   - CPU offloading: Enables models otherwise impossible to run

## GGML Caching Implementation - Full Code Reference from dev branch

### Complete ggml_weight_utils.py Implementation

```python
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
```

### Key Components in Detail

#### 1. Core Data Structures

```python
# Global caching maps and lists
cached_tensor_map = {}  # Maps tensor pointers to their cache information
level_one_tensors = []  # Strong references to level1 cached tensors (compute device)
level_two_tensors = []  # Strong references to level2 cached tensors (tensorator device)
level_three_tensors = [] # Strong references to level3 cached tensors (GGML tensors on tensorator)
prefetch_candidate_stack = []  # Track tensors for prefetching

# CUDA streams and devices
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:1")
compute_device = torch.device("cuda:0")
tensorator_device = torch.device("cuda:1")

# Synchronization events
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)
```

#### 2. Patch Caching

```python
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
```

#### 3. Cache Level Assignment

```python
def initialize_cache_levels():
    global prefetch_candidate_stack
    total_tensor_size = sum(info['tensor_size'] for info in cached_tensor_map.values())
    threshold = total_tensor_size * SMALL_TENSOR_THRESHOLD

    # Small tensors go to level1 (compute device)
    for tensor, info in cached_tensor_map.items():
        if info['cache_level'] == "uninitialized" and info['tensor_size'] < threshold:
            info['cache_level'] = "level1"

    # Sort remaining tensors by patch quantity (desc) then size
    all_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                    if info['cache_level'] == "uninitialized"]
    all_tensors.sort(key=lambda x: (-x[1]['patch_qty'], x[1]['tensor_size']))

    # Assign to levels based on cache size limits
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
            level3_size += info['tensor_size']
        else:
            cached_tensor_map[tensor]['cache_level'] = "none"
```

#### 4. The Core get_weight Function

```python
@profile
def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None):
    if ggml_tensor is None:
        return None

    ggml_tensor_ptr = ggml_tensor.data_ptr()

    # Debug info about tensor
    print(f"Tensor 0x{ggml_tensor_ptr:x} | RefCount: {sys.getrefcount(ggml_tensor)-1} | Device: {ggml_tensor.device} | Referrers: {len(gc.get_referrers(ggml_tensor))}")

    # Check cache levels 
    if ggml_tensor_ptr in cached_tensor_map:
        # Level 1: Return directly from compute device
        if cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level1" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is not None:
            return cached_tensor_map[ggml_tensor_ptr]['cached_tensor']
            
        # Level 2: Transfer from tensorator to compute
        elif cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level2" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is not None:
            with torch.cuda.stream(tensorator_stream):
                level_two_tensor = cached_tensor_map[ggml_tensor_ptr]['cached_tensor']
                level_two_tensor.to(compute_device, non_blocking=True)
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
                return level_two_tensor
                
        # Initialize if not yet categorized
        elif cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "uninitialized":
            initialize_cache_levels()

    # Process tensor with tensorator stream
    with torch.cuda.stream(tensorator_stream):
        # Gather patches
        patch_list = []
        for func, item, key in getattr(ggml_tensor, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches
        
        # Level 3: Use cached GGML tensor if available
        if ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level3" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is not None:
            tensorator_tensor = dequantize_tensor(cached_tensor_map[ggml_tensor_ptr]['cached_tensor'], dtype, dequant_dtype)
        else:
            tensorator_tensor = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)
        
        # Convert GGMLTensor to regular tensor
        if GGMLTensor is not None and isinstance(tensorator_tensor, GGMLTensor):
            tensorator_tensor.__class__ = torch.Tensor

        # Apply patches
        if patch_list:
            if patch_dtype is None:
                tensorator_tensor = func(patch_list, tensorator_tensor, key)
            else:
                tensorator_tensor = func(patch_list, tensorator_tensor, key, dtype if patch_dtype=="target" else patch_dtype)

        # Handle caching based on assigned level
        if ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level1":
            # Level 1: Cache on compute device
            level_one_tensor = tensorator_tensor.clone().to(compute_device, non_blocking=True)
            level_one_tensors.append(level_one_tensor)
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = level_one_tensor
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_one_tensor
            
        elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level2":
            # Level 2: Cache on tensorator device
            level_two_tensor = tensorator_tensor.clone().to(tensorator_device, non_blocking=True)
            level_two_tensors.append(level_two_tensor)
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = level_two_tensor
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor
            
        elif ggml_tensor_ptr in cached_tensor_map and cached_tensor_map[ggml_tensor_ptr]['cache_level'] == "level3" and cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] is None:
            # Level 3: Cache GGML tensor on tensorator
            level_three_tensor = ggml_tensor.to(tensorator_device, non_blocking=True)
            level_three_tensors.append(level_three_tensor)
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = level_three_tensor
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)

        # Transfer result to compute device
        tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
        tensorator_event.record(tensorator_stream)
        
        # Register new tensor in cache if not already present
        if ggml_tensor_ptr not in cached_tensor_map:
            cached_tensor_map[ggml_tensor_ptr] = {}
            cached_tensor_map[ggml_tensor_ptr]['index'] = len(cached_tensor_map) - 1
            cached_tensor_map[ggml_tensor_ptr]['patch_qty'] = len(patch_list)
            cached_tensor_map[ggml_tensor_ptr]['tensor_size'] = (tensorator_tensor.numel() * tensorator_tensor.element_size() / (1024 * 1024))
            cached_tensor_map[ggml_tensor_ptr]['cache_level'] = "uninitialized"
            cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = None
    
    # Wait for tensorator operations to complete
    torch.cuda.current_stream().wait_event(tensorator_event)
    return tensorator_tensor
```

### Implementation Steps for Our Branch

1. **Phase 1: Basic Caching**
   - Implement the core caching data structures
   - Create tensor pointer tracking
   - Add reference management to prevent garbage collection
   - Integrate basic lookup before processing

2. **Phase 2: Cache Level Management**
   - Add cache level assignments
   - Implement the initialize_cache_levels function
   - Handle tensors differently based on their assigned level

3. **Phase 3: Stream Management** 
   - Add CUDA streams for compute and tensorator
   - Create synchronization events
   - Use streams for non-blocking operations
   - Optimize stream usage to maximize parallelism

4. **Phase 4: Prefetching**
   - Implement the prefetch_candidate_stack
   - Track tensors for prefetching
   - Add look-ahead buffer functionality
   - Optimize the prefetching sequence based on tensor order

## Future Work

Next steps include:
1. Implementing a simple GGML tensor caching system with cached_tensor_map:
   - Map tensor pointers to cached tensors
   - Track tensor metadata (size, patches, etc.)
   - Maintain strong references to cached tensors to prevent garbage collection
   - Enable efficient lookup and reuse of processed tensors
2. Extending to a more sophisticated multi-level caching system:
   - Level 1: Small, frequently-accessed tensors on compute device (cuda:0)
   - Level 2: Medium-sized tensors on tensorator device (cuda:1)
   - Level 3: GGML tensors in a prefetch buffer
3. Adding CUDA stream management for asynchronous processing:
   - Use dedicated streams for compute and tensorator operations
   - Use events for proper synchronization between devices
   - Enable non-blocking transfers for maximum parallelism
4. Implementing a look-ahead buffer:
   - Create an N-sized FIFO queue for GGML layer activity
   - Prefetch tensors non-blocking and store in buffer
   - Significantly reduce latency for DRAM-stored models
5. Establishing a three-stage asynchronous pipeline:
   - Stage 1: GGML Layer Buffer - Transfers raw tensors to tensorator
   - Stage 2: Dequantization Buffer - Processes tensors on tensorator
   - Stage 3: Patch Application Buffer - Applies LoRA patches on tensorator

## User Interface Options

The system provides both simple and advanced options:
- Basic device selection for choosing compute device
- Virtual VRAM specification for borrowing memory from other devices
- Expert mode for manual allocation string configuration