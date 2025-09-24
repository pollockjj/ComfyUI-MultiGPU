# Technical Context & Dependencies

## Core Technology Stack

### Python Environment
**Requirements**:
- **Python 3.8+**: ComfyUI minimum requirement
- **PyTorch 2.0+**: Core tensor operations and device management
- **CUDA 11.8+/12.x**: GPU compute support (when available)
- **ComfyUI**: Host framework (dynamic dependency)

### Framework Dependencies

#### Required (ComfyUI Core)
```python
import torch
import comfy.model_management as mm
import comfy.utils
import folder_paths
```

#### Optional (External Custom Nodes)
```python
# ComfyUI-GGUF Integration
try:
    from ComfyUI_GGUF import nodes as gguf_nodes
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

# WanVideoWrapper Integration  
try:
    import ComfyUI_WanVideoWrapper.nodes as wanvideo_nodes
    WANVIDEO_AVAILABLE = True
except ImportError:
    WANVIDEO_AVAILABLE = False
```

## Device Support Matrix

### Primary Support (Tested)
- **CUDA**: GeForce RTX series, Professional/Quadro cards
- **CPU**: x86_64 systems with sufficient RAM (16GB+ recommended)
- **MPS**: Apple Silicon (M1/M2/M3) via Metal Performance Shaders

### Extended Support (Community Validated)
- **XPU**: Intel Arc GPUs, integrated graphics
- **NPU**: Intel NPU for Core 7 processors
- **HIP/ROCm**: AMD GPUs on Linux (community contributed)
- **DirectML**: Windows ML acceleration layer

### Hardware Constraints

#### Memory Requirements
- **Minimum RAM**: 16GB system memory
- **Recommended RAM**: 32GB+ for large model offloading
- **VRAM**: No minimum (CPU-only operation supported)
- **Storage**: NVMe SSD recommended for model loading speed

#### Connection Bandwidth Hierarchy
1. **NVLINK 2x3090**: 50.8 GB/s (optimal)
2. **PCIe 5.0 x16**: ~63 GB/s theoretical (future GPUs)
3. **PCIe 4.0 x16**: ~27.2 GB/s measured
4. **PCIe 3.0 x16**: ~15.8 GB/s theoretical  
5. **PCIe 3.0 x8**: ~6.8 GB/s measured
6. **PCIe 3.0 x4**: ~2.1 GB/s measured

## Development Environment

### Supported Operating Systems
- **Linux**: Primary development platform (Ubuntu 20.04+, others)
- **Windows 10/11**: Full support with CUDA/DirectML
- **macOS**: MPS support for Apple Silicon

### Development Tools
- **IDE**: VSCode with Python extensions
- **Version Control**: Git with conventional commits
- **Testing**: Manual validation across hardware configurations
- **Performance**: Built-in benchmarking tools

### Build System
```toml
# pyproject.toml
[build-system]
requires = ["setuptools", "wheel"]

[project]
name = "comfyui-multigpu"
version = "2.4.7"
dependencies = []  # All dependencies via ComfyUI
```

## Integration Architecture

### ComfyUI Core Integration Points

#### Model Management Hooks
```python
# Patch ComfyUI's device management
original_get_torch_device = mm.get_torch_device
original_text_encoder_device = mm.text_encoder_device

def get_torch_device_patched():
    return current_multigpu_device or original_get_torch_device()
```

#### Node Registration System
```python
# Dynamic node creation based on available dependencies
NODE_CLASS_MAPPINGS = {}

# Core MultiGPU nodes (always available)
for node_name in ["UNETLoader", "VAELoader", "CLIPLoader"]:
    if node_name in GLOBAL_NODE_CLASS_MAPPINGS:
        NODE_CLASS_MAPPINGS[f"{node_name}MultiGPU"] = override_class(
            GLOBAL_NODE_CLASS_MAPPINGS[node_name]
        )

# Conditional nodes based on extensions
if GGUF_AVAILABLE:
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorch2MultiGPU"] = create_gguf_distorch_node()
```

### External Custom Node Integrations

#### ComfyUI-GGUF
- **Purpose**: GGUF quantized model support
- **Integration**: DisTorch for layer-wise distribution
- **Requirements**: city96/ComfyUI-GGUF installed
- **Nodes Created**: 6 GGUF-specific MultiGPU nodes

#### ComfyUI-WanVideoWrapper
- **Purpose**: Kijai's optimized video model support  
- **Integration**: BlockSwap + MultiGPU device selection
- **Requirements**: kijai/ComfyUI-WanVideoWrapper installed
- **Nodes Created**: 8 WanVideo-specific MultiGPU nodes

#### ComfyUI-Florence2
- **Purpose**: Microsoft Florence2 vision model support
- **Integration**: Standard MultiGPU device override
- **Requirements**: kijai/ComfyUI-Florence2 installed
- **Nodes Created**: 2 Florence2-specific MultiGPU nodes

## Performance Characteristics

### Memory Transfer Patterns

#### Optimal Configurations
```python
OPTIMAL_CONFIGS = {
    "image_generation": {
        "priority": "bandwidth",
        "recommended": ["nvlink", "pcie_4_0_x16_cpu"],
        "acceptable": ["pcie_3_0_x16_cpu"],
        "avoid": ["pcie_3_0_x8_gpu", "pcie_3_0_x4_gpu"]
    },
    "video_generation": {
        "priority": "capacity", 
        "recommended": ["any_available"],
        "acceptable": ["pcie_3_0_x4_gpu", "slow_cpu"],
        "avoid": []
    }
}
```

#### Transfer Optimization
- **Pinned Memory**: CPU→GPU transfers use pinned memory allocation
- **Non-blocking Transfers**: GPU→GPU uses asynchronous copying
- **Batch Transfers**: Multiple small layers combined into single transfer
- **Memory Pressure**: Automatic garbage collection during heavy usage

### Model-Specific Behaviors

#### GGUF Models (DisTorch V1/V2)
- **Quantization**: Q8_0, Q6_K, Q4_K_M supported
- **Layer Granularity**: Individual GGML tensor distribution
- **Performance**: 10% speed improvement in DisTorch V2
- **Memory**: Native quantized storage, no dequantization overhead

#### SafeTensor Models (DisTorch V2)
- **Precision**: FP16, BF16, FP8 native support
- **LoRA Compatibility**: Full-precision patching on compute device
- **Layer Distribution**: Based on tensor memory footprint
- **Quality**: No quality loss vs single-GPU operation

## Configuration Management

### Expert Allocation String Formats

#### Bytes Mode (Recommended)
```python
# Format: "device1,amount1;device2,amount2;overflow_device,*"
BYTES_EXAMPLES = [
    "cuda:0,2.5gb;cpu,*",                    # Simple CPU offload
    "cuda:0,500mb;cuda:1,3.0g;cpu,*",       # Multi-GPU distribution  
    "cuda:0,1024mb;cuda:1,2048mb;cpu,*"     # Exact memory control
]
```

#### Ratio Mode (llama.cpp style)
```python  
# Format: "device1,percentage1%;device2,percentage2%"
RATIO_EXAMPLES = [
    "cuda:0,25%;cpu,75%",        # 1:3 split
    "cuda:0,40%;cuda:1,60%",     # GPU-only distribution
    "cuda:0,10%;cuda:1,10%;cpu,80%"  # Multi-device split
]
```

#### Legacy Fraction Mode
```python
# Format: fraction of device VRAM to use
FRACTION_EXAMPLES = [
    0.8,    # Use 80% of available VRAM
    0.5,    # Use 50% of available VRAM  
    0.95    # Use 95% of available VRAM
]
```

## Development Constraints

### ComfyUI API Stability
- **Challenge**: ComfyUI core evolves rapidly
- **Strategy**: Minimal API surface area, fail-loudly on changes
- **Pattern**: Use inheritance to adapt to API evolution
- **Testing**: Validate against multiple ComfyUI versions

### Hardware Diversity  
- **Challenge**: Thousands of possible hardware combinations
- **Strategy**: Focus on most common configurations
- **Community**: User-contributed validation for edge cases
- **Benchmarking**: Systematic performance characterization

### Memory Management Complexity
- **Challenge**: PyTorch + CUDA memory semantics
- **Strategy**: Leverage ComfyUI's existing memory management
- **Safety**: Automatic fallbacks for allocation failures
- **Monitoring**: Built-in memory pressure detection

## Debugging & Monitoring

### Logging Infrastructure
```python
import logging
logger = logging.getLogger("MultiGPU")

# Structured logging for performance analysis
logger.info(f"[DisTorch2] Model {model_id} allocated: {allocation_summary}")
logger.debug(f"Layer {layer_name} transferred {source} -> {target} in {transfer_time}ms")
```

### Performance Telemetry
- **Transfer Times**: Track layer transfer latencies
- **Memory Usage**: Monitor VRAM/RAM utilization per device
- **Model Loading**: Time model initialization phases
- **Inference Impact**: Measure per-step slowdown vs baseline

### Error Categories
1. **Device Detection**: Missing GPUs, driver issues
2. **Memory Allocation**: OOM, fragmentation problems  
3. **Model Loading**: Corrupt files, missing dependencies
4. **Integration**: ComfyUI API changes, extension conflicts

## Future Technology Considerations

### Next-Generation Hardware
- **PCIe 5.0**: 63 GB/s bandwidth capability
- **NVLink 4.0**: 112.5 GB/s for future GPUs
- **DDR5**: Higher memory bandwidth for CPU offloading
- **CXL Memory**: Unified memory pool architectures

### Emerging Platforms
- **Intel Arc**: XPU support expanding
- **AMD RDNA**: HIP/ROCm improvements
- **ARM64**: Apple Silicon and server adoption
- **Distributed**: Multi-node inference possibilities

### Model Architecture Evolution
- **Mixture of Experts**: Sparse model support
- **Multimodal**: Vision+Language combined models
- **Streaming**: Real-time model serving requirements
- **Quantization**: Advanced formats (FP4, INT8, block-wise)
