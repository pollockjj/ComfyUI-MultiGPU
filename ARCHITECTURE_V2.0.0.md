# ComfyUI-MultiGPU Architecture V2.0.0
## DisTorch SafeTensor - Block Swap Memory Management

### ⚠️ BOOTSTRAP DOCUMENT - START HERE AFTER CONTEXT RESET ⚠️

**PURPOSE**: This document captures the exact understanding and implementation plan for DisTorch SafeTensor, which generalizes the block-swap concept from WanVideoWrapper for any SafeTensor model.

**STATUS**: Implemented. The logic has been integrated into `__init__.py` with robust block detection and detailed logging.

---

## STEP 1: UNDERSTAND THE EXISTING APPROACHES

### 1A. ComfyUI-GGUF DisTorch Implementation
**File**: `ComfyUI-GGUF/__init__.py`
**Mechanism**: Distributes individual quantized layers across multiple devices. Dequantizes layers just-in-time for computation. Optimized for maximum memory savings with GGUF models. Provides detailed, formatted logging tables.

### 1B. ComfyUI-WanVideoWrapper Block Swap
**File**: `ComfyUI-WanVideoWrapper/nodes_model_loading.py`
**Mechanism**: Swaps entire, pre-defined model blocks between a compute device and a swap device. It is highly effective but tailored specifically for the WanVideo model architecture.

### 1C. ComfyUI-MultiGPU Integration
**File**: `ComfyUI-MultiGPU/__init__.py`
**IMPLEMENTATION LOCATION**: All DisTorch logic is implemented within this single file to ensure portability and avoid external dependencies.

---

## STEP 2: THE EXACT PROBLEM WE'RE SOLVING

Users have large SafeTensor models (like SDXL, FLUX, etc.) that do not fit into a single GPU's VRAM. The goal is to provide a memory management solution that is flexible, model-agnostic, and provides clear, informative logging, on par with the GGUF DisTorch implementation.

**DisTorch SafeTensor (NEW)**: A memory management solution that intelligently discovers and swaps large, contiguous blocks of a model between a primary compute GPU and a secondary swap device (another GPU or system RAM).

---

## STEP 3: DisTorch SafeTensor IMPLEMENTATION SPEC

### The Wrapper Function
The core of the implementation is the `override_class_with_distorch_safetensor` function, which wraps existing ComfyUI model loaders.

### UI Parameters (What Users See)
The node provides four key parameters to control the memory swapping behavior, ordered for intuitive use:

1.  **`compute_device`**: The primary GPU where computations will occur (e.g., `cuda:0`).
2.  **`compute_reserved_swap_gb`**: The amount of VRAM (in GB) to keep reserved on the `compute_device` for active blocks. This acts as a hot-cache.
3.  **`virtualram_swap_device`**: The device to offload inactive blocks to (e.g., `cpu` or `cuda:1`).
4.  **`virtualram_gb`**: The total size (in GB) of model blocks to offload to the `virtualram_swap_device`.

### Intelligent Block Discovery
The `apply_block_swap` function now uses a multi-stage process to find swappable blocks, ensuring compatibility with various model architectures, including FLUX.
1.  **Standard UNet Structure**: Checks for `input_blocks`, `middle_block`, `output_blocks`.
2.  **Generic `blocks` Attribute**: Looks for a `model.blocks` or `diffusion_model.blocks` list.
3.  **Generic `layers` Attribute**: Looks for a `model.layers` or `diffusion_model.layers` list.
4.  **Fallback to ModuleList**: Scans for any top-level `torch.nn.ModuleList` as a last resort.

### High-Quality Datalogging
A new `analyze_safetensor_distorch` function generates detailed, formatted tables in the console, identical in style to the GGUF implementation, showing:
-   Device Allocations
-   Block Analysis (Type, Count, Memory)
-   Final Block Assignments

---

## STEP 4: CURRENT IMPLEMENTATION CODE IN __init__.py

The following code is a representation of the current implementation within `__init__.py`.

```python
def analyze_safetensor_distorch(model, compute_device, swap_device, virtual_vram_gb, reserved_swap_gb, all_blocks):
    """Provides a detailed analysis of the block swap configuration, mimicking the GGUF DisTorch style."""
    # ... (Full implementation in __init__.py)

def apply_block_swap(model_patcher, compute_device="cuda:0", swap_device="cpu",
                    virtual_vram_gb=4.0, reserved_swap_gb=1.0):
    """
    Applies WanVideo-style block swapping by patching the forward method of individual model blocks.
    """
    # ... (Full implementation with intelligent block discovery in __init__.py)
```

---

## STEP 5: REGISTRATION IN __init__.py

The DisTorch SafeTensor wrappers are registered for all relevant core ComfyUI nodes under the `...DisTorchMultiGPU` alias.

```python
# Register the new DisTorch SafeTensor wrappers
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleDisTorchMultiGPU"] = override_class_with_distorch_safetensor(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["UNETLoaderDisTorchMultiGPU"] = override_class_with_distorch_safetensor(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
# ... and so on for VAELoader, CLIPLoader, ControlNetLoader, etc.
```

---

## STEP 6: KEY DIFFERENCES

### DisTorch (GGUF)
- **Granularity**: Per-layer.
- **Logging**: High-quality, formatted tables.
- **Use Case**: Maximum memory saving on GGUF models.

### DisTorch SafeTensor (NEW)
- **Granularity**: Per-block (intelligently discovered).
- **Logging**: High-quality, formatted tables (matches GGUF style).
- **Use Case**: Balancing memory and speed for **any** SafeTensor model.

### WanVideo Block Swap
- **Granularity**: Per-block (model-specific).
- **Logging**: Basic.
- **Use Case**: Optimized specifically for WanVideo models.

---

## STEP 7: IMPLEMENTATION CHECKLIST

- [X] Delete `blockswap.py` (DONE)
- [X] Document the approach (THIS DOCUMENT)
- [X] Implement `override_class_with_distorch_safetensor` in `__init__.py` (DONE)
- [X] Rename and reorder UI parameters (DONE)
- [X] Expand coverage to all core ComfyUI nodes (DONE)
- [X] **Implement robust, multi-stage block discovery to support models like FLUX (DONE)**
- [X] **Overhaul logging to match the high-quality, formatted style of the GGUF DisTorch implementation (DONE)**
- [ ] Test with SDXL checkpoint
- [ ] Test with Flux checkpoint
- [ ] Verify memory usage matches expectations
- [ ] Measure transfer overhead

---

## FUTURE EXTENSIONS

1.  **Auto-mode**: Automatically determine optimal settings based on available VRAM and model size.
2.  **Dynamic Block Sizing**: Group layers into blocks dynamically instead of relying on the model's predefined block structure.
3.  **Advanced Profiling**: Add tools to measure transfer overhead and help users optimize their settings.
