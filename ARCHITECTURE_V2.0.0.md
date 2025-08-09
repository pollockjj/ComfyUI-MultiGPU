# ComfyUI-MultiGPU Architecture V2.0.0
## DisTorch SafeTensor - Block Swap Memory Management

### ⚠️ BOOTSTRAP DOCUMENT - START HERE AFTER CONTEXT RESET ⚠️

**PURPOSE**: This document captures the exact understanding and implementation plan for DisTorch SafeTensor, which generalizes the block-swap concept from WanVideoWrapper for any SafeTensor model.

**STATUS**: Implemented. The logic has been integrated into `__init__.py`.

---

## STEP 1: UNDERSTAND THE EXISTING APPROACHES

### 1A. ComfyUI-GGUF DisTorch Implementation
**File**: `ComfyUI-GGUF/__init__.py` and `gguf_model_patcher.py`
**Mechanism**: Distributes individual quantized layers across multiple devices. Dequantizes layers just-in-time for computation. Optimized for maximum memory savings with GGUF models.

### 1B. ComfyUI-WanVideoWrapper Block Swap
**File**: `ComfyUI-WanVideoWrapper/nodes_model_loading.py`
**Mechanism**: Swaps entire, pre-defined model blocks (e.g., ResNet blocks, Attention blocks) between a compute device and a swap device during inference. It is highly effective but tailored specifically for the WanVideo model architecture.

### 1C. ComfyUI-MultiGPU Integration
**File**: `ComfyUI-MultiGPU/__init__.py`
**IMPLEMENTATION LOCATION**: All DisTorch logic is implemented within this single file to ensure portability and avoid external dependencies.

---

## STEP 2: THE EXACT PROBLEM WE'RE SOLVING

Users have large SafeTensor models that do not fit into a single GPU's VRAM. We provide them with a solution that is more flexible than single-layer offloading and more general-purpose than WanVideo's integrated approach.

**DisTorch SafeTensor (NEW)**: A memory management solution that intelligently swaps large, contiguous blocks of a model between a primary compute GPU and a secondary swap device (another GPU or system RAM).

---

## STEP 3: DisTorch SafeTensor IMPLEMENTATION SPEC

### The Wrapper Function
The core of the implementation is the `override_class_with_distorch_safetensor` function, which wraps existing ComfyUI model loaders.

```python
def override_class_with_distorch_safetensor(cls):
    """DisTorch wrapper for SafeTensor models, providing block-swap memory optimization."""
```

### UI Parameters (What Users See)
The node provides four key parameters to control the memory swapping behavior, ordered for intuitive use:

1.  **`compute_device`**: The primary GPU where computations will occur (e.g., `cuda:0`).
2.  **`compute_reserved_swap_gb`**: The amount of VRAM (in GB) to keep reserved on the `compute_device` for active blocks. This acts as a hot-cache.
3.  **`virtualram_swap_device`**: The device to offload inactive blocks to (e.g., `cpu` or `cuda:1`).
4.  **`virtualram_gb`**: The total size (in GB) of model blocks to offload to the `virtualram_swap_device`. This effectively creates "virtual VRAM" on your compute device.

### The Math (20GB Model Example)
- **Model**: 20GB total size.
- **`compute_device`**: `cuda:0` (24GB VRAM)
- **`compute_reserved_swap_gb`**: `1.0` GB
- **`virtualram_swap_device`**: `cpu`
- **`virtualram_gb`**: `4.0` GB

**Result**:
- **4GB** of the model's blocks are immediately moved to the `cpu`.
- The remaining **16GB** of blocks are loaded onto `cuda:0`.
- During inference, blocks are swapped as needed, but a buffer of at least **1GB** (`compute_reserved_swap_gb`) worth of blocks is kept on the compute device if possible.

### Operation Flow
1.  The user selects a model using a DisTorch-wrapped loader (e.g., `CheckpointLoaderSimpleDisTorchMultiGPU`).
2.  The model is loaded normally by the underlying ComfyUI loader.
3.  The `apply_block_swap` function analyzes the model to identify swappable blocks (e.g., input, middle, and output blocks of a UNet).
4.  Based on `virtualram_gb`, a number of blocks are moved to the `virtualram_swap_device`.
5.  The `forward` method of each offloaded block is patched with a hook.
6.  When the model runs, the hook moves the required block to the `compute_device` just before it's needed and moves it back to the `virtualram_swap_device` afterward, using non-blocking transfers for efficiency.

---

## STEP 4: CURRENT IMPLEMENTATION CODE IN __init__.py

The following code is a representation of the current implementation within `__init__.py`.

```python
def override_class_with_distorch_safetensor(cls):
    """DisTorch wrapper for SafeTensor models, providing block-swap memory optimization."""
    
    class NodeOverrideDisTorchSafeTensor(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            compute_device = devices[1] if len(devices) > 1 else devices[0]
            
            inputs["optional"] = inputs.get("optional", {})
            
            # Reordered and renamed parameters
            inputs["optional"]["compute_device"] = (devices, {
                "default": compute_device,
                "tooltip": "Primary device for computation."
            })
            inputs["optional"]["compute_reserved_swap_gb"] = ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
                "max": 16.0,
                "step": 0.1,
                "tooltip": "GB of VRAM to keep reserved on the compute device."
            })
            inputs["optional"]["virtualram_swap_device"] = (devices, {
                "default": "cpu",
                "tooltip": "Device to offload inactive model blocks to."
            })
            inputs["optional"]["virtualram_gb"] = ("FLOAT", {
                "default": 4.0,
                "min": 0.1,
                "max": 64.0,
                "step": 0.1,
                "tooltip": "Amount of VRAM (in GB) to offload to the swap device."
            })
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, compute_device=None, compute_reserved_swap_gb=1.0, 
                     virtualram_swap_device="cpu", virtualram_gb=4.0, **kwargs):
            global current_device
            
            logging.info(f"[DisTorch SafeTensor] Override called with: compute_device={compute_device}, swap_device={virtualram_swap_device}, virtualram_gb={virtualram_gb}, reserved_gb={compute_reserved_swap_gb}")

            if compute_device is not None:
                current_device = compute_device
            
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            model = out[0]
            if hasattr(model, 'model'):
                logging.info("[DisTorch SafeTensor] Model has 'model' attribute, applying block swap.")
                apply_block_swap(
                    model,
                    compute_device=compute_device,
                    swap_device=virtualram_swap_device,
                    virtual_vram_gb=virtualram_gb,
                    reserved_swap_gb=compute_reserved_swap_gb
                )
            else:
                logging.warning("[DisTorch SafeTensor] Loaded object does not have a 'model' attribute, skipping block swap.")
            
            return out

    return NodeOverrideDisTorchSafeTensor


def apply_block_swap(model_patcher, compute_device="cuda:0", swap_device="cpu",
                    virtual_vram_gb=4.0, reserved_swap_gb=1.0):
    """
    Applies WanVideo-style block swapping by patching the forward method of individual model blocks.
    """
    # ... (Full implementation in __init__.py)
```

---

## STEP 5: REGISTRATION IN __init__.py

The new DisTorch SafeTensor wrappers are registered for all relevant core ComfyUI nodes.

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
- **Use Case**: Maximum memory saving on GGUF models, often with CPU offload.
- **Implementation**: Complex allocation strings and quantization handling.

### DisTorch SafeTensor (NEW)
- **Granularity**: Per-block.
- **Use Case**: Balancing memory and speed for any SafeTensor model.
- **Implementation**: Simple forward hooks, model-agnostic.

### WanVideo Block Swap
- **Granularity**: Per-block (model-specific).
- **Use Case**: Optimized specifically for WanVideo models.
- **Implementation**: Integrated directly into the custom model's forward pass.

---

## WHY THIS MATTERS

1.  **Flexibility**: Enables running models that are larger than a single GPU's VRAM.
2.  **Control**: Users can fine-tune the memory vs. speed trade-off.
3.  **Compatibility**: Works with all standard SafeTensor models loaded through core ComfyUI nodes.
4.  **Simplicity**: All logic is self-contained within the `ComfyUI-MultiGPU` custom node.

---

## STEP 7: IMPLEMENTATION CHECKLIST

- [X] Delete `blockswap.py` (DONE)
- [X] Document the approach (THIS DOCUMENT)
- [X] Implement `override_class_with_distorch_safetensor` in `__init__.py` (DONE)
- [X] Rename and reorder UI parameters (DONE)
- [X] Expand coverage to all core ComfyUI nodes (DONE)
- [ ] Test with SDXL checkpoint
- [ ] Test with Flux checkpoint
- [ ] Verify memory usage matches expectations
- [ ] Measure transfer overhead

---

## FUTURE EXTENSIONS

1.  **Auto-mode**: Automatically determine optimal settings based on available VRAM and model size.
2.  **Dynamic Block Sizing**: Group layers into blocks dynamically instead of relying on the model's predefined block structure.
3.  **Advanced Profiling**: Add tools to measure transfer overhead and help users optimize their settings.
