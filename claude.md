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

## Future Work

Next steps include:
1. Adding quick-and-dirty patch caching 
2. Implementing a GGML look-ahead buffer for DRAM-stored models
3. Creating non-blocking transfers to pinned CPU memory
4. Establishing a pipeline buffer for level 3 tensors

## User Interface Options

The system provides both simple and advanced options:
- Basic device selection for choosing compute device
- Virtual VRAM specification for borrowing memory from other devices
- Expert mode for manual allocation string configuration