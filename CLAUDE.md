# ComfyUI-MultiGPU Development Guide

This guide captures the key architecture, decisions, and implementation details for ComfyUI-MultiGPU, specifically focusing on performance optimizations using dual GPUs.

## Working Philosophy

> "I work in steps not leaps. I know this is hard for you so that is why we are having this conversation. I will likely redirect you to do smaller and smaller chunks until I can understand what you are doing. The truth? I wrote none of this alone. I used llms for every line. That said, I wrote ALL of it, because I took the 1000s of lines of code and extracted only what was necessary in as elegant a manner as I am capable. Your code will not survie this. Our code will. I am a test engineer. I write optimal code = only what is necessary."

This philosophy guides our development approach - focused on iterative steps, understanding each component, extracting only what's necessary, and ensuring optimal efficiency in the final code.

## Project Architecture

ComfyUI-MultiGPU implements three key strategies for enhancing memory management:

1. **DisTorch Virtual VRAM**: Offloads UNet layers to free up VRAM on the main compute GPU
2. **Multi-GPU Processing**: Distributes workloads across multiple GPUs
3. **Optimized Caching**: Uses advanced buffer management to maximize throughput

### Key Files

- `__init__.py`: Core module initialization and node registration, implements DisTorch for distributing model layers across devices
- `nodes.py`: Node implementations for MultiGPU features
- `ggml_weight_utils.py`: GPU memory management and tensor operations, implements the core pipeline

## Performance Optimization Strategy

Our current focus is optimizing the GGML model processing pipeline by creating an efficient system that:

1. Keeps GGML layers on their natural device (as DisTorch can place them anywhere)
2. Performs dequantization and LoRA patching on the tensorator (cuda:1)
3. Handles compute operations on the compute device (cuda:0)
4. Uses multi-level caching and ping-pong buffers to maximize throughput

### Tensor Processing Pipeline Design

The pipeline works with the natural flow of data rather than forcing unnecessary transfers. Based on profiling data, we've determined that explicit tensor transfers at the beginning are inefficient. Instead:

1. We leave tensors on their loaded device (compute or other if using DisTorch)
2. Process patches on the tensorator, where they remain during inference
3. Dequantize and patch tensors on the tensorator device
4. Transfer the fully processed result back to compute for inference

This approach maximizes available VRAM on the compute device for latent space operations, which is critical for applications like HunyuanVideo and Wan Video.

### Performance Split Target

- Dequantization & LoRA application: 25% of time (tensorator)
- Compute operations: 75% of time (compute)

### Key Memory Insights

- Dequantized tensors are ephemeral, currently recreated during each inference step
- LoRA application must happen during every dequantization for memory efficiency
- Patches remain on tensorator device during inference, preserving compute VRAM

## Key Implementation Concepts

### 1. Multi-Level Caching System

We implement a sophisticated caching system with multiple levels:

- **Level Zero Cache**: High-priority tensors stored directly on compute device
- **Level Two Cache**: Tensors stored on tensorator device with weakrefs
- **Ping-Pong Buffers**: Two rotating buffers to enable prefetching
- **Patch Cache**: Efficiently reuse patch data across operations

```python
# Core ping-pong mechanism
active_buffer_index = 0  # 0 = buffer A, 1 = buffer B
prefetch_buffers = [{}, {}]  # Two buffer dictionaries
prefetch_batch_size = 15  # Size of each buffer
```

### 2. CUDA Stream Management

We use dedicated CUDA streams for asynchronous operations across devices:

```python
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:1")
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)
```

### 3. DisTorch Flexibility

A key insight: DisTorch allows GGML layers to be placed on any device, not just compute. This enables advanced optimizations but also requires careful handling of tensor locations. The get_weight function adapts to the actual location of tensors rather than assuming they start on the compute device.

## Implementation Plan

### Phase 1: Basic Linear Pipeline (COMPLETED)
Our first step was to implement a simple, linear pipeline that:
1. Loads initial GGML tensors on compute (no change from current behavior)
2. Transfers tensors to tensorator
3. Performs dequantization on tensorator
4. Applies patches (LoRA) on tensorator
5. Transfers processed tensors back to compute for inference
6. Returns the fully processed weight

Based on profiling, we eliminated the explicit transfer in step 2, as it was more efficient to process tensors in their natural location.

#### Phase 1 Performance Results
Performance profiling of the basic implementation:
- 206 seconds for 8 LoRAs (93.4% time spent in LoRA application)
- 24.3 seconds for one LoRA
- 18.7 seconds for zero LoRAs

These results confirm that LoRA application is the dominant bottleneck, especially with multiple LoRAs, validating our approach of offloading this work to tensorator.

### Phase 2: Targeted Optimizations (CURRENT FOCUS)
Now that we've proven the core concept works, we're focusing on targeted optimizations based on profiling results:

1. **Optimizations for Multiple LoRAs**:
   - Combine multiple LoRA patches before applying them
   - Parallelize LoRA applications where possible
   - Use streams effectively to overlap computation

2. **Optimizations for Limited Memory**:
   - Implement GGML-Layer buffered cache using slower memory (DRAM) 
   - Add smart memory limit detection
   - Implement adaptive chunking based on available memory

3. **Transfer Optimizations**:
   - Investigate using DMA for transfers rather than explicit .to() operations
   - Optimize specifically for hardware configurations with NVLink
   - Experiment with different synchronization patterns

### Phase 3: Advanced Pipeline (FUTURE)
Once the targeted optimizations are complete, we'll implement the full advanced pipeline:
- Complete ping-pong buffer system with continuous refilling
- 30-layer chunking for efficient memory use
- Minimized synchronization points
- Advanced telemetry for performance monitoring

## Implementation Checklist

### Phase 1 (COMPLETED)
- [x] Add tensor caching configuration
- [x] Add NVTX profiling markers
- [x] Implement basic ping-pong buffer system
- [x] Create basic linear pipeline
- [x] Eliminate inefficient explicit tensor transfers
- [x] Add UI option to toggle tensorator processing
- [x] Gather initial performance metrics

### Phase 2 (CURRENT FOCUS)
- [ ] Optimize for multiple LoRAs:
  - [ ] Implement LoRA batch processing
  - [ ] Create combined LoRA patch application
  - [ ] Add stream management for overlapped operations
- [ ] Optimize memory management:
  - [ ] Build GGML-Layer buffered cache
  - [ ] Implement adaptive chunking based on available memory
  - [ ] Add OOM prevention mechanisms
- [ ] Optimize transfer mechanisms:
  - [ ] Test DMA vs explicit transfers
  - [ ] Optimize for NVLink configurations
  - [ ] Minimize synchronization overhead

### Phase 3 (FUTURE)
- [ ] Implement the full advanced pipeline:
  - [ ] Complete continuous ping-pong buffer system 
  - [ ] Implement 30-layer chunking mechanism
  - [ ] Minimize synchronization points
  - [ ] Add detailed performance telemetry

## Key Code Patterns

### Processing Tensors on Tensorator

The updated approach processes tensors directly in their current location without forcing explicit transfers:

```python
# Keep GGML layer on loaded device (compute or other if DisTorch)
tensorator_device = torch.device("cuda:1")
with torch.cuda.stream(tensorator_stream):
    # Prepare patches on tensorator - patches remain here during inference
    patch_list = []
    for func, item, key in getattr(tensor, "patches", []):
        patches = retrieve_cached_patch(item, tensorator_device, key)
        patch_list += patches
    
    # Dequantize tensor directly (ephemeral tensor, recreated each inference)
    w = dequantize_tensor(tensor, dtype, dequant_dtype)
    
    # Apply patches on tensorator
    if patch_list:
        w = func(patch_list, w, key)
        
    # Transfer result back to compute
    w = w.to(device="cuda:0", non_blocking=True)
```

### Multi-Level Caching System

The implementation uses a sophisticated caching mechanism with multiple tiers:

```python
if ptr in cached_tensor_map and "level_zero_cache_location" in cached_tensor_map[ptr]:
    return cached_tensor_map[ptr]["level_zero_cache_location"]

# Check ping-pong buffers
buf = prefetch_buffers[active_buffer_index]
if ptr in buf:
    # ... buffer management logic ...
    return buf[ptr]

# Check level two cache
if ptr in cached_tensor_map and "level_two_cache_location" in cached_tensor_map[ptr]:
    with torch.cuda.stream(tensorator_stream):
        w = cached_tensor_map[ptr]["level_two_cache_location"]().clone()
    # ... prefetch scheduling logic ...
    return w
```

## Future Enhancement Ideas

1. **GGML-Layer Buffered Cache**: Store GGML layers in DRAM and buffer them to tensorator, optimizing for both memory and performance

2. **Full Tensorator Utilization**: Keep the tensorator VRAM filled with cached, fully-patched tensors ready for use, with continuous refilling to ensure the pipeline is never empty

3. **DMA Transfer Optimization**: Investigate whether letting compute pull tensors from tensorator via DMA is more efficient than explicit transfers, especially with NVLink hardware

4. **Combined LoRA Application**: For multiple LoRAs, combine patches before applying to reduce computational overhead:
   ```python
   super_patch = (alpha1 * LoRA1) + (alpha2 * LoRA2) + ... + (alpha8 * LoRA8)
   weight += super_patch
   ```

## Profiling Commands

For CUDA profiling:
```bash
nsys profile --stats=true python -m comfy.run_engine --config comfy/config.yaml
```

For function profiling:
```python
@profile
def function_to_profile():
    # code here
```

## Links and Resources

- [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)