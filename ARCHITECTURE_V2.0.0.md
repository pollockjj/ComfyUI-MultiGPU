# ComfyUI-MultiGPU Architecture V2.0.0

## Executive Summary

ComfyUI-MultiGPU provides intelligent model distribution across multiple GPUs and system RAM, optimizing for minimal VRAM usage while maintaining performance. Version 2.0 introduces a unified interface supporting both layer-by-layer transfers (DisTorch) and block swapping strategies.

## Core Concepts

### 1. Virtual VRAM
Virtual VRAM represents the extended memory pool available by offloading model components to other devices (CPU RAM or secondary GPUs). The system manages transfers between devices transparently during inference.

### 2. Transfer Strategies

#### DisTorch (Layer-by-Layer)
- **Mechanism**: Spoofs quantized tensors on offload device, dequantizes JIT to compute device
- **Transfer Size**: Single layer at a time (~100-200MB)
- **VRAM Usage**: Minimal (1 layer active)
- **Best For**: Video generation (long inference times)
- **Trade-off**: Many small PCIe transfers

```python
# DisTorch approach - minimal VRAM footprint
def forward_hook(module, input, output):
    # Load single layer
    load_layer_to_device(module, compute_device)
    output = module.forward(input)
    # Immediately offload
    offload_layer(module, offload_device)
    return output
```

#### Block Swap (New in V2)
- **Mechanism**: Moves blocks of layers between devices
- **Transfer Size**: Configurable (1-8GB blocks)
- **VRAM Usage**: Reserved swap buffer
- **Best For**: Image generation (short inference times)
- **Trade-off**: Fewer, larger PCIe transfers

```python
# Block swap approach - batched transfers
def forward_hook(module, input, output):
    if need_swap(module):
        # Swap entire block
        offload_block(current_block, offload_device)
        load_block(next_block, compute_device)
    return module.forward(input)
```

### 3. Unified Interface

All strategies share common parameters:
```python
class VirtualVRAMConfig:
    virtual_vram_gb: float    # Total model size to offload
    swap_space_gb: float      # Reserved buffer on compute device
    swap_device: str          # Where to offload ("cpu", "cuda:1")
    
    # Derived behavior
    if swap_space_gb < min_layer_size:
        use_distorch()  # Layer-by-layer
    else:
        use_block_swap()  # Block transfers
```

## Implementation Architecture

### Memory Management

#### Size Calculation (Shared Utility)
```python
def calculate_model_size(model):
    """Calculate actual memory footprint"""
    total_bytes = 0
    for param in model.parameters():
        if hasattr(param, 'quant_type'):  # GGUF
            # Account for quantization
            total_bytes += calculate_gguf_size(param)
        else:  # Safetensor
            total_bytes += param.element_size() * param.nelement()
    return total_bytes / (1024**3)  # GB
```

#### Block Partitioning
```python
def partition_model(model, swap_space_gb):
    """Divide model into swappable blocks"""
    blocks = []
    current_block = []
    current_size = 0
    
    for name, module in model.named_modules():
        module_size = get_module_size(module)
        
        if current_size + module_size > swap_space_gb:
            # Start new block
            blocks.append(current_block)
            current_block = [module]
            current_size = module_size
        else:
            current_block.append(module)
            current_size += module_size
    
    return blocks
```

### Hook System

#### Pre/Post Forward Hooks
```python
class ModelWrapper:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.blocks = partition_model(model, config.swap_space_gb)
        self.current_block_idx = -1
        
        # Install hooks
        for block_idx, block in enumerate(self.blocks):
            for module in block:
                module.register_forward_pre_hook(
                    lambda m, i: self.pre_forward(m, block_idx)
                )
    
    def pre_forward(self, module, block_idx):
        if block_idx != self.current_block_idx:
            # Swap blocks
            self.swap_blocks(self.current_block_idx, block_idx)
            self.current_block_idx = block_idx
```

### GGUF Handling

#### Quantized Tensor Management
```python
class GGUFHandler:
    def handle_gguf_layer(self, layer):
        if self.config.swap_space_gb < layer.size:
            # Use DisTorch approach - dequantize JIT
            return self.distorch_dequantize(layer)
        else:
            # Can move entire quantized block
            return self.block_swap_quantized(layer)
    
    def distorch_dequantize(self, layer):
        """Dequantize during transfer (COPY operation)"""
        # Creates new tensor on compute device
        return dequantize_to_device(layer, self.compute_device)
    
    def block_swap_quantized(self, layer):
        """Move quantized tensor (SWAP operation)"""
        # Moves existing tensor between devices
        return layer.to(self.compute_device)
```

## Phase Implementation Plan

### Phase 1: Block Swap for Safetensors (Current Focus)

**Goal**: Implement configurable block swapping for non-quantized models.

**Implementation**:
```python
class DisTorchBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "virtual_vram_gb": ("FLOAT", {
                    "default": 4.0, 
                    "min": 0.1, 
                    "max": 64.0,
                    "step": 0.1
                }),
                "swap_space_gb": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 16.0,
                    "step": 0.1
                }),
                "swap_device": (["cpu", "cuda:0", "cuda:1"],),
            }
        }
    
    def apply(self, model, virtual_vram_gb, swap_space_gb, swap_device):
        # Calculate model size
        model_size = calculate_model_size(model)
        
        # Partition into blocks
        blocks = partition_model(model, swap_space_gb)
        
        # Install swap hooks
        wrapper = BlockSwapWrapper(model, blocks, swap_device)
        
        return (wrapper.model,)
```

### Phase 2: Unified GGUF Support

**Goal**: Extend block swap to GGUF models, auto-selecting strategy.

**Decision Logic**:
```python
def select_strategy(model, config):
    if is_gguf(model):
        min_layer = get_min_layer_size(model)
        if config.swap_space_gb < min_layer:
            return DisTorchStrategy()  # Must dequantize JIT
        else:
            return BlockSwapStrategy()  # Can move quantized blocks
    else:
        return BlockSwapStrategy()  # Safetensors always use blocks
```

### Phase 3: Auto-Optimization

**Goal**: Use empirical data to auto-configure optimal settings.

See `DOE_OPTIMIZATION.md` for detailed benchmarking plan.

**Auto Mode**:
```python
def auto_configure(model, workload):
    # Detect hardware
    pcie_gen = detect_pcie_generation()
    gpu_bandwidth = detect_gpu_bandwidth()
    
    # Analyze workload
    is_video = workload.frames > 1
    latent_size = workload.height * workload.width
    
    # Lookup optimal config from DOE results
    if is_video:
        return {"swap_space_gb": 0.1}  # Minimize transfers
    else:
        return lookup_optimal_config(
            model.size, latent_size, pcie_gen
        )
```

## Performance Characteristics

### Transfer Overhead Analysis

| Strategy | Transfer Size | Frequency | PCIe Time | Best Case |
|----------|--------------|-----------|-----------|-----------|
| DisTorch | 100-200MB | Every layer | High | Video (long inference) |
| Block Swap (1GB) | 1GB | Every ~10 layers | Medium | Balanced |
| Block Swap (4GB) | 4GB | Every ~40 layers | Low | Image (short inference) |

### Memory Usage Patterns

```
DisTorch (0.1GB swap):
|===|                    <- Active layer (100MB)
|...|...|...|...|...|   <- Offloaded layers

Block Swap (2GB swap):
|==========|            <- Active block (2GB)
|..........|..........|  <- Offloaded blocks
```

## Advantages Over Existing Solutions

### vs. Sequential CPU Offload
- **Granular Control**: Configure exact offload amount
- **Multi-GPU Support**: Use secondary GPUs as fast swap
- **Quantization Aware**: Handles GGUF efficiently

### vs. Model Parallelism
- **No Model Modification**: Works with any model
- **Dynamic**: Adjusts to available resources
- **Flexible**: User controls memory/speed trade-off

## Code Organization

```
ComfyUI-MultiGPU/
├── nodes.py              # Node definitions
├── core/
│   ├── distorch.py      # Original layer-by-layer
│   ├── blockswap.py     # New block swapping
│   ├── memory.py        # Shared memory utilities
│   └── hooks.py         # Hook management
├── strategies/
│   ├── auto.py          # Auto-optimization
│   ├── gguf.py          # GGUF-specific handling
│   └── safetensor.py    # Safetensor handling
└── benchmark/
    ├── doe.py           # DOE test runner
    └── profiles.py      # Hardware profiles
```

## Testing Strategy

### Unit Tests
- Memory calculation accuracy
- Block partitioning logic
- Hook installation/removal

### Integration Tests
- Safetensor models (SDXL, Flux)
- GGUF models (quantized)
- Multi-GPU configurations

### Performance Tests
- Measure transfer overhead
- Verify memory usage
- Benchmark vs baseline

## Migration Path

### For Existing Users
1. Current DisTorch nodes continue working
2. New unified node available alongside
3. Gradual migration as benefits proven

### Configuration Migration
```python
# Old DisTorch
distorch_model = DisTorch(model, device_map)

# New Unified (equivalent)
unified_model = VirtualVRAM(
    model, 
    virtual_vram_gb=model_size,
    swap_space_gb=0.1,  # DisTorch-like
    swap_device="cpu"
)
```

## Future Directions

### Adaptive Strategies
- Monitor transfer patterns
- Adjust block size dynamically
- Predict optimal points

### Pipeline Integration
- Coordinate with samplers
- Batch-aware swapping
- Multi-model orchestration

### Hardware Acceleration
- Direct Storage API
- NVLink optimization
- CXL memory pooling

## Conclusion

Version 2.0 unifies memory management strategies under a coherent interface, providing users with fine-grained control over the memory/performance trade-off while maintaining backward compatibility and preparing for future optimizations.
