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

## GGML Caching Implementation

The advanced tensor caching system is now implemented in the ggml_weight_utils_dev.py file in the dev branch, which has been committed to the repository. This implementation features:

1. A sophisticated three-level caching strategy
2. CUDA stream management with event synchronization
3. Tensor tracking by pointer address
4. Reference preservation to prevent garbage collection
5. Level-based tensor management for optimal memory usage

## Next Implementation Task: Early Tensor Mapping

We now need to focus on constructing an improved `cached_tensor_map` starting at the module loading phase rather than waiting for first inference. This will provide several advantages:

1. **Complete tensor information available immediately**: All tensors will be cataloged before first inference
2. **Hard references maintained from load**: Prevent garbage collection by maintaining references from load time
3. **Optimal caching decisions at startup**: Pre-calculate caching levels before any inference occurs
4. **Deterministic behavior**: Make the system more predictable and consistent

The implementation will build a comprehensive tensor map structure with the following properties for each tensor:

```python
cached_tensor_map[tensor_ptr] = {
    'index': index_number,              # Unique sequential index
    'name': full_module_path,           # Full path to module (for debugging/analysis)
    'load_device': assigned_device,     # Device where tensor is loaded
    'patch_qty': patch_count,           # Number of patches for this tensor
    'tensor_size': size_in_mb,          # Size of tensor in MB
    'cache_level': cache_assignment,    # Cache level (uninitialized, level1, level2, level3, none)
    'cached_tensor': tensor_reference   # Reference to cached processed tensor
}
```

## Implementation Steps

1. **Phase 1: Load-Time Tensor Tracking**
   - Track tensors as they're loaded during model initialization
   - Store module name, device, and tensor metadata
   - Maintain hard references to prevent garbage collection

2. **Phase 2: Cache Level Assignment**
   - Implement sizing and categorization at load completion
   - Assign levels based on size, patch count, and usage patterns
   - Pre-allocate caching structures for commonly used tensors

3. **Phase 3: Stream Management**
   - Setup CUDA streams for asynchronous operations
   - Configure synchronization points between operations
   - Optimize stream usage to maximize parallelism

4. **Phase 4: Prefetching**
   - Add deterministic prefetching based on tensor access patterns
   - Implement look-ahead buffer functionality
   - Optimize the prefetching sequence based on tensor order

## Implementation Location

The critical implementation point is in `patched_load` function:
1. When modules are assigned to devices with `module_object.to(target_device_for_module)`
2. After the entire model is moved with `self.model.to(device_to)`

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

## CRITICAL INSTRUCTIONS FOR CLAUDE

### Implementation Requirements

1. **PRECISE CODE FOLLOWING**
   - NEVER modify existing code structure
   - Match EXACTLY the coding style of the project
   - Make ONLY the specific changes requested
   - NEVER "optimize" or "improve" working code
   - NO extra functionality, comments, or "helpful" additions

2. **IMPLEMENTATION LOCATIONS**
   - Tensor tracking MUST happen in patched_load during assignment
   - Key points are when flat_assignments is populated (DisTorch path)
   - And when model.to(device_to) is called (non-DisTorch path)
   - NEVER modify get_weight or core functions without explicit instruction

3. **CODING STYLE REQUIREMENTS**
   ```python
   # ALWAYS follow this exact structure:
   cached_tensor_map[ggml_tensor_ptr] = {}
   cached_tensor_map[ggml_tensor_ptr]['index'] = len(cached_tensor_map) - 1
   cached_tensor_map[ggml_tensor_ptr]['name'] = full_name
   cached_tensor_map[ggml_tensor_ptr]['load_device'] = target_device
   cached_tensor_map[ggml_tensor_ptr]['patch_qty'] = len(patches_data)
   cached_tensor_map[ggml_tensor_ptr]['tensor_size'] = tensor_size_mb
   cached_tensor_map[ggml_tensor_ptr]['cache_level'] = "uninitialized"
   cached_tensor_map[ggml_tensor_ptr]['cached_tensor'] = None
   ```

4. **PHASED IMPLEMENTATION**
   - Phase 1: Only implement tensor tracking in patched_load
   - Phase 2: Implement prefetching using tracked indices (only when requested)
   - Phase 3: Add multi-level caching (only when requested)
   - Phase 4: Add stream management (only when requested)

IMPORTANT: NEVER proceed to the next phase without explicit direction.

## User Interface Options

The system provides both simple and advanced options:
- Basic device selection for choosing compute device
- Virtual VRAM specification for borrowing memory from other devices
- Expert mode for manual allocation string configuration