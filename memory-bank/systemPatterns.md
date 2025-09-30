# System Architecture & Patterns (Updated 2025-09-29)

## Core Architecture

### Dynamic Class Override System
Foundation Pattern: City96's elegant inheritance-based approach (Dec 2024 revolution)

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

Key Benefits:
- 50 lines vs 400+: Eliminated manual class definitions
- Universal Support: Works with any ComfyUI loader node
- Maintenance: Auto-adapts to ComfyCore changes
- Consistency: Unified behavior across all MultiGPU nodes

### Load-Patch-Distribute (LPD) Method
DisTorch2 Core Process:

```python
# 1. LOAD - Always on compute device first
tensor = load_tensor_on_compute_device(tensor_name)

# 2. PATCH - Apply all LoRAs at full precision
if lora_patches:
    tensor = apply_lora_patches(tensor, lora_patches, precision=torch.float16)

# 3. DISTRIBUTE - Move to target device after patching
final_tensor = tensor.to(target_device)
```

Design Principles:
- Quality First: No precision loss during LoRA application
- Deterministic: Same allocation every time
- ComfyUI Native: Works with existing ComfyCore patterns

## Memory Management Architecture

### Virtual VRAM System
Concept: Make CPU/secondary GPU memory appear as extended VRAM

```python
class VirtualVRAM:
    def __init__(self, compute_device, donor_device, virtual_gb):
        self.compute_device = compute_device  # e.g., "cuda:0"
        self.donor_device = donor_device      # e.g., "cpu" or "cuda:1"
        self.virtual_gb = virtual_gb          # Extended memory pool
        
    def allocate_layers(self, model_layers, allocation_string):
        # "cuda:0,2.5gb;cpu,*" -> assign layers based on cumulative memory
```

### Expert Allocation Modes

Bytes Mode (Recommended):
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

Ratio Mode (llama.cpp style):
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

### Selective Ejection Pipeline (v2.5.0 - VERIFIED WORKING)

**Load-time Flagging** (per-model transient):
```python
# In DisTorch2 wrapper after real loader returns
if hasattr(out[0], 'model') and hasattr(out[0].model, '_mgpu_keep_loaded'):
    keep_loaded = out[0].model._mgpu_keep_loaded
    out[0].model._mgpu_unload_distorch_model = (not keep_loaded)
```
Purpose: Mark specific DisTorch models for ejection when user unchecks "keep loaded"

**Manager-Parity Cleanup Trigger**:
```python
def force_full_system_cleanup(reason="manual", force=True):
    pq.set_flag("unload_models", True)  # Exactly what Manager's
    pq.set_flag("free_memory", True)    # "Free model and node cache" does
```

**Selective Unloading** (patched `mm.unload_all_models`):
```python
def _mgpu_patched_unload_all_models():
    # Categorize models by flag
    models_to_unload = [lm for lm in mm.current_loaded_models 
                        if getattr(lm.model, '_mgpu_unload_distorch_model', False)]
    kept_models = [lm for lm in mm.current_loaded_models
                   if not getattr(lm.model, '_mgpu_unload_distorch_model', False)]
    
    if kept_models:
        # Selective unload: eject flagged, retain others
        for lm in models_to_unload:
            lm.model_unload(unpatch_weights=True)
        
        # Add GC anchors to prevent premature collection
        for lm in kept_models:
            add_retention_anchor(lm.model, "keep_loaded_protection")
        
        # Rebuild with kept models only
        mm.current_loaded_models = kept_models
    else:
        # No models to keep - standard cleanup
        _mgpu_original_unload_all_models()
```

**Multi-Device VRAM + CPU Management** (patched `mm.soft_empty_cache`):
```python
def soft_empty_cache_distorch2_patched(force=False):
    # 1. Detect DisTorch2 activity
    is_distorch_active = any(model_hash in safetensor_allocation_store 
                            for model in mm.current_loaded_models)
    
    # 2. VRAM allocator management
    if is_distorch_active:
        soft_empty_cache_multigpu()  # Clear all device caches
    else:
        original_soft_empty_cache(force)  # Standard single-device
    
    # 3. Adaptive CPU memory management
    check_cpu_memory_threshold()
    
    # 4. Forced executor reset (Manager parity)
    if force:
        trigger_executor_cache_reset(reason="forced_soft_empty", force=True)
```

**Verified Working** (Production Logs 2025-09-30):
```
[CATEGORIZE_SUMMARY] kept_models: 2, models_to_unload: 1, total: 3
[SELECTIVE_UNLOAD] Proceeding with selective unload: retaining 2, unloading 1
[UNLOAD_EXECUTE] Unloading model: Flux
[REMAINING_MODEL] 0: AutoencodingEngine  
[REMAINING_MODEL] 1: FluxClipModel_
```

### Device Detection & Management

Multi-Device Enumeration:
```python
def get_device_list():
    devices = ["cpu"]  # Always available
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    # XPU/NPU/MLU/MPS/DirectML/CoreX detection...
    return devices
```

Device Bandwidth Intelligence (from benchmarking):
1. NVLINK (~50.8 GB/s)
2. PCIe 4.0 x16 (~27.2 GB/s)
3. PCIe 3.0 x8 (~6.8 GB/s)
4. PCIe 3.0 x4 (~2.1 GB/s)

## Integration Patterns

### ComfyCore Alignment
Philosophy: Work WITH ComfyUI, not against it

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
```

### Dependency Detection
```python
def check_module_availability(module_paths):
    for path in module_paths:
        if os.path.exists(os.path.join(custom_nodes_dir, path)):
            return True
    return False
```

## Performance Optimization Patterns

### Layer Transfer Optimization
```python
def optimized_layer_transfer(layer, source_device, target_device):
    if source_device == target_device:
        return layer
    non_blocking = "cuda" in source_device and "cuda" in target_device
    if source_device == "cpu" and "cuda" in target_device:
        layer = layer.pin_memory()
    return layer.to(target_device, non_blocking=non_blocking)
```

### Memory Pressure Management
```python
def should_auto_offload(model_size_gb, vram_available_gb, threshold=0.9):
    return model_size_gb > (vram_available_gb * threshold)

def calculate_offload_amount(model_size_gb, target_vram_usage_gb):
    return max(0, model_size_gb - target_vram_usage_gb)
```

## Error Handling Philosophy

### Fail Loudly Pattern
```python
# GOOD: Let ComfyCore changes surface immediately
def load_model(self, model_name, device):
    return original_loader.load_unet(model_name, device)

# AVOID: Defensive coding that masks issues
try:
    return original_loader.load_unet(model_name, device)
except AttributeError:
    return fallback_method()
```

### Integration Validation
```python
def validate_comfycore_integration():
    required_attrs = ['FUNCTION', 'INPUT_TYPES', 'RETURN_TYPES']
    for attr in required_attrs:
        if not hasattr(target_class, attr):
            raise AttributeError(f"ComfyCore node missing {attr} - API changed")
```

## Code Style Patterns

### Self-Documenting Code
```python
def override_class_with_device_selection(original_class, device_param_name="device"):
    compute_device = kwargs.get(device_param_name, mm.get_torch_device())
```

### Minimal Comments Philosophy
Prefer structure and naming to convey intent; use comments for non-obvious constraints/assumptions.

## Architectural Decision Records

### Why Dynamic Class Override vs Manual Definitions
Decision: Use inheritance-based class override (City96 approach)
Rationale:
- Reduces code from 400+ lines to ~50 lines
- Auto-adapts to ComfyCore changes
- Eliminates maintenance burden of manual node definitions
- Provides consistent behavior across all node types

### Why Load-Patch-Distribute vs Direct Distribution
Decision: Always load on compute device first, then distribute
Rationale:
- Ensures LoRA patches applied at full precision
- Maintains quality parity with single-GPU workflows
- Predictable behavior regardless of target device
- Works with ComfyCore’s existing patching mechanisms

### Why Expert Modes vs Automatic Only
Decision: Provide both automatic and expert allocation modes
Rationale:
- Automatic mode enables low-VRAM users immediately
- Expert modes allow optimization for specific hardware
- Performance depends on bandwidth topology; experts need control

### Why Universal Device Support vs CUDA-Only
Decision: Support CPU, XPU, NPU, MLU, MPS, DirectML alongside CUDA
Rationale:
- ComfyUI’s user base spans diverse hardware
- Future-proof for emerging accelerators
- Hardware democracy principle

### Why Per-Model Flag vs Global Sentinel (Updated)
Decision: Use per-model `_mgpu_unload_distorch_model` instead of a global “DISTORCH2_UNLOAD_MODEL” sentinel
Rationale:
- Surgical precision at model granularity
- No persistent or cross-workflow state
- Cleaner semantics under ComfyUI’s queue/flag model

Hardened unloading rule (target to re-apply):
- If no models are flagged for ejection, `mm.unload_all_models` must be a strict no-op to preserve retained models across the full Manager-parity flow.

## Testing & Validation Patterns

### Hardware Configuration Testing
```python
HARDWARE_CONFIGS = [
    {"compute": "cuda:0", "donor": "cpu", "connection": "PCIe 4.0 x16"},
    {"compute": "cuda:0", "donor": "cuda:1", "connection": "NVLink"},
    {"compute": "cuda:0", "donor": "cuda:1", "connection": "PCIe 3.0 x8"},
    {"compute": "cuda:0", "donor": "cuda:1", "connection": "PCIe 3.0 x4"},
]
```

### Model Compatibility Validation
```python
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
    baseline_time = benchmark_single_gpu(model)
    for allocation in allocation_configs:
        distributed_time = benchmark_distributed(model, hardware_config, allocation)
        performance_ratio = distributed_time / baseline_time
        assert performance_ratio < expected_slowdown_threshold(hardware_config)
```

## Recent Refactorings (v2.5.0)

### DisTorch2 Allocation Consolidation (-179 lines)
**Problem**: 85% code duplication between `analyze_safetensor_loading()` and `analyze_safetensor_loading_clip()`

**Solution**: Unified function with CLIP support flag
```python
def _extract_clip_head_blocks(raw_block_list, compute_device):
    """Helper: Identify and pre-assign CLIP head blocks to compute device"""
    head_keywords = ['embed', 'wte', 'wpe', 'token_embedding', 'position_embedding']
    head_blocks = []
    distributable_blocks = []
    block_assignments = {}
    
    for module_size, module_name, module_object, params in raw_block_list:
        if any(kw in module_name.lower() for kw in head_keywords):
            head_blocks.append((module_size, module_name, module_object, params))
            block_assignments[module_name] = compute_device
        else:
            distributable_blocks.append((module_size, module_name, module_object, params))
    
    return head_blocks, distributable_blocks, block_assignments, head_memory

def analyze_safetensor_loading(model_patcher, allocations_string, is_clip=False):
    """Unified allocation function with CLIP head preservation support"""
    # Common allocation logic...
    
    if is_clip:
        head_blocks, distributable_raw, block_assignments, head_memory = \
            _extract_clip_head_blocks(raw_block_list, compute_device)
        # Adjust compute_device quota for head blocks
        donor_quotas[compute_device] -= head_memory
    else:
        distributable_raw = raw_block_list
        block_assignments = {}
    
    # Continue with unified distribution logic...
```

**Benefits**:
- Single source of truth for allocation
- CLIP special case isolated in 20-line helper
- Easier to maintain and debug
- Same behavior, cleaner architecture

### Production Cleanup (-40 lines)
**Removed**: Diagnostic instrumentation wrapper `_mgpu_instrumented_soft_empty_cache()`

**Rationale**: Pure debug logging with no production function - removed to clean codebase

**Result**: Clear separation between device_utils.py (functional) and model_management_mgpu.py (lifecycle)

## Module Architecture (Post-Refactoring)

### Core Module Separation
Problem Solved: Eliminated circular import `device_utils.py` ↔ `distorch_2.py`

Solution: `model_management_mgpu.py` as central model lifecycle hub

### Module Responsibilities

device_utils.py (Base Layer):
- Device enumeration and detection
- VRAM cache management (`soft_empty_cache_multigpu`)
- Pure hardware abstraction – no model tracking

model_management_mgpu.py (Core Layer):
- Model lifecycle tracking and memory logging
- Cleanup orchestration (`force_full_system_cleanup`, `trigger_executor_cache_reset`, `check_cpu_memory_threshold`)
- Patched unload path (selective ejection)

distorch_2.py/distorch.py (Feature Layer):
- DisTorch distribution algorithms and allocation analysis
- Per-model flagging (`_mgpu_unload_distorch_model`) during DisTorch loads
- Imports FROM Core/Base only

UI Layer: nodes.py, checkpoint_multigpu.py
- Device-aware user interfaces and node definitions

Assembly: __init__.py
- Final integration/patch registration (`mm.soft_empty_cache` patch, node maps)

### Import Flow Architecture
```
    ┌─────────────────┐
    │    __init__.py  │ ← Assembly Layer
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │  UI Layer       │ ← nodes.py, checkpoint_multigpu.py
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │ Feature Layer   │ ← distorch_2.py, distorch.py
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │ Core Layer      │ ← model_management_mgpu.py
    └─────────────────┘
           ↑
    ┌─────────────────┐
    │ Base Layer      │ ← device_utils.py  
    └─────────────────┘
```

### Architectural Validation
Rule: Dependencies only flow UPWARD. Violations create circular imports.

Prevention: Before any import, verify it respects the layer hierarchy.

### Function Migration Record
Moved from device_utils.py to model_management_mgpu.py:
- `multigpu_memory_log` – memory state logging
- `trigger_executor_cache_reset` – CPU memory management
- `check_cpu_memory_threshold` – adaptive cleanup triggers
- `force_full_system_cleanup` – Manager-parity free flow

Rationale: These belong to model lifecycle/cleanup, not hardware enumeration.
