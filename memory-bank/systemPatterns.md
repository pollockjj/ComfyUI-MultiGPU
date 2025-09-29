# System Architecture & Patterns

## Core Architecture

### Dynamic Class Override System
**Foundation Pattern**: City96's elegant inheritance-based approach (Dec 2024 revolution)

```python
def override_class(original_class, device_param="device"):
    class MultiGPUClass(original_class):
        @classmethod
        def INPUT_TYPES(cls):
            inputs = original_class.INPUT_TYPES()
            inputs["required"][device_param] = (get_device_list(),)
            return inputs
            
        def override(self, *args, **kwargs):
            device = kwargs.pop(device_param, None)
            mm.text_encoder_device = device
            return original_class.FUNCTION(self, *args, **kwargs)
            
    return MultiGPUClass
```

**Key Benefits**:
- **50 lines vs 400+**: Eliminated manual class definitions
- **Universal Support**: Works with any ComfyUI loader node
- **Maintenance**: Auto-adapts to ComfyCore changes
- **Consistency**: Unified behavior across all MultiGPU nodes

### Load-Patch-Distribute (LPD) Method
**DisTorch2 Core Process**:

```python
# 1. LOAD - Always on compute device first
tensor = load_tensor_on_compute_device(tensor_name)

# 2. PATCH - Apply all LoRAs at full precision
if lora_patches:
    tensor = apply_lora_patches(tensor, lora_patches, precision=torch.float16)

# 3. DISTRIBUTE - Move to target device after patching
final_tensor = tensor.to(target_device)
```

**Design Principles**:
- **Quality First**: No precision loss during LoRA application
- **Deterministic**: Same allocation every time
- **ComfyUI Native**: Works with existing ComfyCore patterns

## Memory Management Architecture

### Virtual VRAM System
**Concept**: Make CPU/secondary GPU memory appear as extended VRAM

```python
class VirtualVRAM:
    def __init__(self, compute_device, donor_device, virtual_gb):
        self.compute_device = compute_device  # e.g., "cuda:0"
        self.donor_device = donor_device      # e.g., "cpu" or "cuda:1"
        self.virtual_gb = virtual_gb          # Extended memory pool
        
    def allocate_layers(self, model_layers, allocation_string):
        # Parse: "cuda:0,2.5gb;cpu,*" 
        # Assign layers based on cumulative memory requirements
```

### Expert Allocation Modes

**Bytes Mode** (Recommended):
```python
# "cuda:0,2.5gb;cuda:1,3.0g;cpu,*"
def parse_bytes_allocation(allocation_string):
    devices = []
    for device_spec in allocation_string.split(';'):
        device_name, memory_spec = device_spec.split(',')
        if memory_spec == '*':
            memory_bytes = float('inf')  # Overflow device
        else:
            memory_bytes = parse_memory_string(memory_spec)  # 2.5gb -> bytes
        devices.append((device_name, memory_bytes))
    return devices
```

**Ratio Mode** (llama.cpp style):
```python
# "cuda:0,25%;cpu,75%" -> 1:3 split
def parse_ratio_allocation(allocation_string):
    total_ratio = sum(float(spec.split(',')[1].rstrip('%')) for spec in allocation_string.split(';'))
    device_ratios = []
    for device_spec in allocation_string.split(';'):
        device_name, ratio_spec = device_spec.split(',')
        ratio = float(ratio_spec.rstrip('%')) / total_ratio
        device_ratios.append((device_name, ratio))
    return device_ratios
```

## Device Detection & Management

### Multi-Device Enumeration
```python
def get_device_list():
    devices = ["cpu"]  # Always available
    
    # CUDA detection
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    
    # Extended device support
    for device_type in ["xpu", "npu", "mlu", "mps"]:
        if device_available(device_type):
            devices.append(device_type)
            
    return devices
```

### Device Bandwidth Intelligence
**Hierarchy** (from benchmarking data):
1. **NVLINK**: ~50.8 GB/s (near-native performance)
2. **PCIe 4.0 x16**: ~27.2 GB/s (excellent CPU offloading)
3. **PCIe 3.0 x8**: ~6.8 GB/s (acceptable for video models)
4. **PCIe 3.0 x4**: ~2.1 GB/s (slow but viable for capacity)

## Integration Patterns

### ComfyCore Alignment
**Philosophy**: Work WITH ComfyUI, not against it

```python
# GOOD: Use ComfyCore's device management
current_device = mm.get_torch_device()
mm.text_encoder_device = target_device

# AVOID: Direct PyTorch device manipulation
torch.cuda.set_device(device_id)  # Bypasses ComfyCore
```

### Node Registration System
```python
# Dynamic registration based on available dependencies
if "ComfyUI-GGUF" in installed_modules:
    NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorch2MultiGPU"] = create_gguf_distorch_node()

if "ComfyUI-WanVideoWrapper" in installed_modules:
    NODE_CLASS_MAPPINGS["WanVideoModelLoaderMultiGPU"] = create_wanvideo_node()
```

### Dependency Detection
```python
def check_module_availability(module_paths):
    for path in module_paths:
        if os.path.exists(os.path.join(custom_nodes_dir, path)):
            return True
    return False

# Example: Check for multiple possible names
GGUF_PATHS = ["ComfyUI-GGUF", "comfyui-gguf", "ComfyUI_GGUF"]
has_gguf = check_module_availability(GGUF_PATHS)
```

## Performance Optimization Patterns

### Layer Transfer Optimization
```python
def optimized_layer_transfer(layer, source_device, target_device):
    """Optimized tensor transfer with memory management"""
    if source_device == target_device:
        return layer
        
    # Use non_blocking for CUDA->CUDA transfers
    non_blocking = "cuda" in source_device and "cuda" in target_device
    
    # Pin memory for CPU->GPU transfers
    if source_device == "cpu" and "cuda" in target_device:
        layer = layer.pin_memory()
        
    return layer.to(target_device, non_blocking=non_blocking)
```

### Memory Pressure Management
```python
def should_auto_offload(model_size_gb, vram_available_gb, threshold=0.9):
    """Automatic offloading when model exceeds 90% of available VRAM"""
    return model_size_gb > (vram_available_gb * threshold)

def calculate_offload_amount(model_size_gb, target_vram_usage_gb):
    """Calculate exact amount to offload for target VRAM usage"""
    return max(0, model_size_gb - target_vram_usage_gb)
```

## Error Handling Philosophy

### Fail Loudly Pattern
```python
# GOOD: Let ComfyCore changes surface immediately
def load_model(self, model_name, device):
    # No try/except - we want to know if ComfyCore changes break us
    return original_loader.load_unet(model_name, device)

# AVOID: Defensive coding that masks issues
try:
    return original_loader.load_unet(model_name, device)
except AttributeError:
    # This hides when ComfyCore API changes
    return fallback_method()
```

### Integration Validation
```python
# Validate ComfyCore compatibility at startup
def validate_comfycore_integration():
    required_attrs = ['FUNCTION', 'INPUT_TYPES', 'RETURN_TYPES']
    for attr in required_attrs:
        if not hasattr(target_class, attr):
            raise AttributeError(f"ComfyCore node missing {attr} - API changed")
```

## Code Style Patterns

### Self-Documenting Code
```python
# GOOD: Names explain purpose
def override_class_with_device_selection(original_class, device_param_name="device"):
    compute_device = kwargs.get(device_param_name, mm.get_torch_device())
    
# AVOID: Cryptic naming requiring comments
def oc_wds(oc, dpn="device"):  # override class with device selection
    cd = kwargs.get(dpn, mm.gtd())  # compute device = get torch device
```

### Minimal Comments Philosophy
```python
# GOOD: Code structure tells the story
class DisTorchLoader:
    def __init__(self, compute_device, donor_device, virtual_vram_gb):
        self.compute_device = compute_device
        self.donor_device = donor_device
        self.virtual_vram_gb = virtual_vram_gb
        
    def load_model_with_distribution(self, model_path, allocation_string):
        model = self.load_on_compute_device(model_path)
        distributed_model = self.distribute_layers(model, allocation_string)
        return distributed_model

# AVOID: Over-commenting obvious code
class DisTorchLoader:
    def __init__(self, compute_device, donor_device, virtual_vram_gb):
        # Set the compute device for processing
        self.compute_device = compute_device
        # Set the donor device for storage
        self.donor_device = donor_device
        # Set the virtual VRAM amount in gigabytes
        self.virtual_vram_gb = virtual_vram_gb
```

## Architectural Decision Records

### Why Dynamic Class Override vs Manual Definitions
**Decision**: Use inheritance-based class override (City96 approach)
**Rationale**: 
- Reduces code from 400+ lines to ~50 lines
- Auto-adapts to ComfyCore changes
- Eliminates maintenance burden of manual node definitions
- Provides consistent behavior across all node types

### Why Load-Patch-Distribute vs Direct Distribution
**Decision**: Always load on compute device first, then distribute
**Rationale**:
- Ensures LoRA patches applied at full precision
- Maintains quality parity with single-GPU workflows
- Predictable behavior regardless of target device
- Works with ComfyCore's existing patching mechanisms

### Why Expert Modes vs Automatic Only
**Decision**: Provide both automatic and expert allocation modes
**Rationale**:
- Automatic mode enables low-VRAM users immediately
- Expert modes allow optimization for specific hardware
- Benchmarking shows performance depends on hardware configuration
- Power users need fine-grained control

### Why Universal Device Support vs CUDA-Only
**Decision**: Support CPU, XPU, NPU, MLU, MPS, DirectML alongside CUDA
**Rationale**:
- ComfyUI runs on diverse hardware platforms
- Apple Silicon (MPS) and Intel hardware (XPU) growing user bases
- Future-proofing for emerging compute devices
- Principle of hardware democracy

## Testing & Validation Patterns

### Hardware Configuration Testing
```python
# Test matrix for different hardware combinations
HARDWARE_CONFIGS = [
    {"compute": "cuda:0", "donor": "cpu", "connection": "PCIe 4.0 x16"},
    {"compute": "cuda:0", "donor": "cuda:1", "connection": "NVLink"},
    {"compute": "cuda:0", "donor": "cuda:1", "connection": "PCIe 3.0 x8"},
    {"compute": "cuda:0", "donor": "cuda:1", "connection": "PCIe 3.0 x4"},
]
```

### Model Compatibility Validation
```python
# Test different model architectures and formats
TEST_MODELS = [
    {"name": "FLUX.1-dev", "format": ".safetensors", "size_gb": 23.8},
    {"name": "WAN 2.2", "format": ".safetensors", "size_gb": 14.0},
    {"name": "FLUX-GGUF", "format": ".gguf", "size_gb": 11.8},
    {"name": "QWEN Image", "format": ".safetensors", "size_gb": 38.0},
]
```

### Performance Regression Testing
```python
def benchmark_allocation_performance(model, hardware_config, allocation_configs):
    """Ensure performance doesn't regress with updates"""
    baseline_time = benchmark_single_gpu(model)
    
    for allocation in allocation_configs:
        distributed_time = benchmark_distributed(model, hardware_config, allocation)
        performance_ratio = distributed_time / baseline_time
        assert performance_ratio < expected_slowdown_threshold(hardware_config)
```

## Module Architecture (Post-Refactoring)

### Core Module Separation
**Problem Solved**: Eliminated circular import `device_utils.py` ↔ `distorch_2.py`

**Solution**: Created `model_management_mgpu.py` as central model lifecycle hub

### Module Responsibilities

**device_utils.py** (Base Layer):
- Device enumeration and detection
- VRAM cache management (`soft_empty_cache_multigpu`)
- Pure hardware abstraction - NO model tracking

**model_management_mgpu.py** (Core Layer): 
- Model lifecycle tracking (`track_modelpatcher`)
- Memory logging (`multigpu_memory_log`) 
- System cleanup (`force_full_system_cleanup`, `trigger_executor_cache_reset`)

**distorch_2.py/distorch.py** (Feature Layer):
- DisTorch distribution algorithms
- SafeTensor/GGUF specific logic
- Imports FROM core/base layers ONLY

### Import Flow Architecture
```
    ┌─────────────────┐
    │    __init__.py  │ ← Assembly Layer
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │  UI Layer       │ ← nodes.py, checkpoint_multigpu.py
    │  (User Interface)│ 
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │ Feature Layer   │ ← distorch_2.py, distorch.py
    │ (DisTorch Logic)│
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │ Core Layer      │ ← model_management_mgpu.py
    │ (Model Lifecycle)│
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │ Base Layer      │ ← device_utils.py  
    │ (Hardware)      │
    └─────────────────┘
```

### Architectural Validation
**Rule**: Dependencies only flow UPWARD. Violations create circular imports.

**Prevention**: Before any import, ask "Does this violate the layer hierarchy?"

### Function Migration Record
**Moved from device_utils.py to model_management_mgpu.py:**
- `multigpu_memory_log` - Memory state logging
- `track_modelpatcher` - ModelPatcher lifecycle tracking  
- `trigger_executor_cache_reset` - CPU memory management
- `check_cpu_memory_threshold` - Adaptive cleanup triggers
- `force_full_system_cleanup` - Full system reset

**Rationale**: These functions manage model lifecycle and memory state, not hardware detection. Separation prevents circular dependencies while maintaining clean responsibilities.
