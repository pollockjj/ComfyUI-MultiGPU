# Product Context: Why ComfyUI-MultiGPU Exists

## The Problem Space

### The VRAM Crisis
Modern AI models are experiencing explosive growth in size:
- **FLUX.1-dev**: 23.8GB (exceeds most consumer cards)
- **WAN 2.2**: 14GB+ (video generation demands)
- **Hunyuan Video**: 25GB+ (next-gen video models)
- **QWEN Image**: Up to 38GB in FP16 (professional image editing)

Meanwhile, consumer hardware remains constrained:
- **RTX 4090**: 24GB VRAM (can't fit largest models)
- **RTX 3090**: 24GB VRAM (aging but still powerful)
- **RTX 4080/4070**: 16GB/12GB (mainstream but limited)
- **Budget Cards**: 8GB or less (significant portion of user base)

### The Workflow Limitation
ComfyUI's default behavior loads entire models onto the primary GPU:
- **Latent space competition**: Model storage vs computation space
- **Resolution limits**: Large models prevent high-resolution generation
- **Batch size restrictions**: Memory consumed by static weights
- **OOM failures**: Workflows simply fail to run

### The Speed vs. Memory Dilemma
Existing solutions force uncomfortable tradeoffs:
- **--lowvram mode**: Dynamic but unpredictable, quality issues with LoRAs
- **Quantization**: Quality loss, limited model support
- **Model switching**: Slow, workflow interruption
- **Single-GPU limitation**: Unused hardware sitting idle

## The Vision

### Unified Compute Pool
Transform multi-GPU setups from "main + unused" to "unified compute":
- **Primary GPU**: 100% dedicated to computation/latent processing
- **Secondary GPUs**: High-speed model storage (NVLINK, PCIe)
- **System RAM**: Extended model storage with optimized transfers
- **Mixed Architectures**: Old cards find new life as storage

### Deterministic Memory Management
Replace dynamic allocation with user-controlled distribution:
- **Static Mapping**: Model layers assigned to specific devices
- **Predictable Performance**: Known transfer costs and timing
- **Quality Preservation**: Full-precision LoRA patching on compute device
- **Workflow Reliability**: Consistent behavior across runs

### Hardware Democracy
Enable AI generation across hardware tiers:
- **Budget Systems**: 8GB card + system RAM for large models
- **Enthusiast Builds**: 2x3090 effectively becomes 48GB unified pool
- **Mixed Setups**: 4090 + old 1080 Ti = expanded capability
- **Enterprise**: Workstation-grade hardware optimization

## User Experience Goals

### For Low-VRAM Users
- **Model Access**: Run any model regardless of VRAM size
- **Resolution Freedom**: Generate at previously impossible dimensions
- **Batch Processing**: Multiple images/frames without OOM
- **Quality Maintenance**: No forced quantization or quality loss

### For Multi-GPU Users
- **Hardware Utilization**: Every GPU contributes meaningfully
- **Performance Optimization**: NVLink, PCIe bandwidth maximization
- **Flexible Distribution**: Fine-grained control over model placement
- **Scaling Benefits**: More hardware = more capability

### For Workflow Creators
- **Predictability**: Consistent memory usage patterns
- **Configurability**: Expert modes for precise control
- **Compatibility**: Works with existing ComfyUI workflows
- **Documentation**: Clear performance expectations

## The Market Reality

### Community Demand
Issues and feedback reveal consistent patterns:
- **"Only cuda:0 visible"**: Multi-GPU setup confusion
- **"Out of memory"**: VRAM exhaustion with large models  
- **"Slow generation"**: Inefficient memory management
- **"Can't run X model"**: Hardware limitations blocking workflows

### Hardware Evolution
Consumer GPU landscape trends:
- **VRAM Stagnation**: 24GB ceiling for years
- **Model Growth**: Exponential size increases
- **Price Pressure**: High-end cards increasingly expensive
- **Mixed Installations**: Users combining new + old hardware

### Ecosystem Position
ComfyUI's role in AI generation:
- **Node-based workflows**: Flexible but memory-hungry
- **Model diversity**: Supports every major architecture
- **Community-driven**: Custom nodes enable specialization
- **Production use**: Professional workflows demand reliability

## Success Metrics

### Technical Success
- **Model Loading**: Any model loads on any hardware combination
- **Performance Predictability**: Benchmarked speed vs. memory tradeoffs
- **Stability**: No crashes or memory leaks in extended use
- **Compatibility**: Works across operating systems and configurations

### User Success  
- **Workflow Enablement**: Previously impossible workflows now work
- **Hardware Investment**: Old GPUs gain new utility
- **Resolution/Batch Scaling**: Tangible output quality improvements
- **Community Growth**: Increasing adoption and positive feedback

### Ecosystem Success
- **ComfyUI Integration**: Seamless operation with core functionality
- **Developer Adoption**: Other custom nodes build on our patterns
- **Hardware Vendor Recognition**: Acknowledged in optimization discussions
- **Production Deployment**: Used in commercial/professional settings
