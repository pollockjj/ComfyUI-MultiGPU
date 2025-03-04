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

- `__init__.py`: Core module initialization and node registration
- `nodes.py`: Node implementations for MultiGPU features
- `ggml_weight_utils.py`: GPU memory management and tensor operations

## Performance Optimization Strategy

Our current focus is optimizing the GGML model processing pipeline by creating an efficient system that:

1. Performs all dequantization and LoRA patching on cuda:1
2. Handles compute operations on cuda:0
3. Uses a chunked pipeline approach to maximize throughput

### GGML Tensor Pipeline Design

The pipeline follows this pattern:
```
GGML Layer (disk/storage) → CUDA:1 (dequant+patch) → CUDA:0 (compute)
```

Specifically:
1. Process layers in 30-layer chunks (~2.2GB)
2. Move chunks to cuda:1 for dequantization and LoRA application
3. Transfer fully processed tensors to cuda:0 for computation

### Performance Split Target

- Dequantization & LoRA application: 25% of time (cuda:1)
- Compute operations: 75% of time (cuda:0)

## Key Implementation Concepts

### 1. Ping-Pong Buffer System

We use a ping-pong buffer mechanism where:
- Buffer A: Active buffer being used for current operations
- Buffer B: Being filled with next batch of processed tensors

```python
# Core ping-pong mechanism
active_buffer_index = 0  # 0 = buffer A, 1 = buffer B
prefetch_buffers = [{}, {}]  # Two buffer dictionaries
```

### 2. CUDA Stream Management

We use dedicated CUDA streams for each device to enable asynchronous operations:

```python
cuda0_stream = torch.cuda.Stream(device="cuda:0")
cuda1_stream = torch.cuda.Stream(device="cuda:1")
cuda0_event = torch.cuda.Event(enable_timing=False)
cuda1_event = torch.cuda.Event(enable_timing=False)
```

### 3. Runtime Configuration

We provide runtime configuration options through the UI:

```python
cache_config = {'use_tensor_cache': False}

# Can be updated dynamically in the UI
cache_config["use_tensor_cache"] = tensor_cache
```

## Implementation Plan

### Phase 1: Basic Linear Pipeline
Our first step is to implement a simple, linear pipeline that:
1. Loads initial GGML tensors on cuda:0 (no change from current behavior)
2. Transfers tensors to cuda:1
3. Performs dequantization on cuda:1
4. Applies patches (LoRA) on cuda:1
5. Transfers processed tensors back to cuda:0 for inference
6. Returns the fully processed weight

This simple approach establishes the foundation without complex buffering or synchronization. It may not be optimally efficient yet, but it verifies the core concept of offloading work to cuda:1.

### Phase 2: Advanced Pipeline (Future)
Once the basic pipeline is proven, we'll enhance it with:
- Ping-pong buffer system for async operation
- 30-layer chunking for efficient memory use
- Minimized synchronization points
- Optimized memory transfers

## Implementation Checklist

### Phase 1 (Current Focus)
- [x] Add tensor caching configuration
- [x] Add NVTX profiling markers
- [x] Implement basic ping-pong buffer system
- [ ] Create basic linear pipeline:
  - [ ] Move GGML tensor to cuda:1
  - [ ] Dequantize on cuda:1
  - [ ] Apply patches on cuda:1
  - [ ] Transfer result back to cuda:0

### Phase 2 (Future)
- [ ] Optimize `get_weight` with buffering
- [ ] Implement 30-layer chunking mechanism
- [ ] Minimize synchronization points
- [ ] Add telemetry for performance monitoring

## Key Code Patterns

### Tensor Dequantization on CUDA:1

```python
# Move tensor to cuda:1
tensor_cuda1 = tensor.to(device="cuda:1", non_blocking=True)

# Dequantize on cuda:1
w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)

# Process any patches on cuda:1
if patch_list:
    w = function(patch_list, w, key)
```

### LoRA Application on CUDA:1

For standard LoRA application, we want to move both operations to cuda:1:

```python
# On cuda:1:
mat1 = v0.to(device=cuda1_device, dtype=intermediate_dtype)
mat2 = v1.to(device=cuda1_device, dtype=intermediate_dtype)
    
# Matrix multiplication on cuda:1
lora_diff = torch.mm(
    mat1.flatten(start_dim=1), 
    mat2.flatten(start_dim=1)
).reshape(weight.shape)
    
# Apply scaling on cuda:1
lora_diff = ((strength * alpha) * lora_diff).to(dtype=output_dtype)

# Apply to weight on cuda:1 
final_weight = weight_cuda1 + lora_diff
```

### Asynchronous Transfer Management

```python
# Record event on main stream
main_event.record(main_stream)
    
# Wait for event on secondary stream
with torch.cuda.stream(secondary_stream):
    secondary_stream.wait_event(main_event)
    # ...process on secondary device...
    secondary_event.record(secondary_stream)
    
# Wait for completion in main stream
main_stream.wait_event(secondary_event)
```

## Future Enhancement Ideas

1. **Combined LoRA Patches**: For multiple LoRA applications, combine patches before applying:
   ```python
   super_patch = (alpha1 * LoRA1) + (alpha2 * LoRA2) + ... + (alpha8 * LoRA8)
   weight += super_patch
   ```

2. **Adaptive Chunking**: Dynamically adjust chunk size based on available memory and model size

3. **Triple Buffering**: Extend to three buffers for more overlap between operations

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