# ComfyUI-MultiGPU Architecture V2.0.0
## DisTorch Unified Interface: SafeTensor & GGUF

### ⚠️ BOOTSTRAP DOCUMENT - START HERE AFTER CONTEXT RESET ⚠️

**PURPOSE**: This document details the standardized, unified DisTorch interface for both SafeTensor and GGUF models, ensuring a consistent user experience.

**STATUS**: Implemented. The logic has been integrated into `__init__.py` with backward compatibility for legacy GGUF workflows.

---

## STEP 1: THE UNIFIED DisTorch PHILOSOPHY

The primary goal is to provide a single, intuitive interface for memory offloading, regardless of the model format. Users should not need to learn two different systems. Both SafeTensor and GGUF DisTorch nodes will now share the exact same UI parameters, leveraging the concepts popularized by the original GGUF implementation.

---

## STEP 2: THE STANDARDIZED DisTorch INTERFACE

Both `override_class_with_distorch_safetensor` and the new `override_class_with_distorch_gguf` will present the following four parameters to the user:

1.  **`compute_device`**: The primary GPU where computations will occur (e.g., `cuda:0`).
2.  **`virtual_ram_gb`**: The amount of "virtual VRAM" to create by offloading parts of the model. This is the primary control for memory savings.
3.  **`donor_device`**: A dropdown list to select a single device to offload to (e.g., `cpu`, `cuda:1`). Defaults to `cpu`.
4.  **`expert_mode_allocations`**: An advanced string for power users to define complex, multi-device distribution schemes, bypassing the simplified controls.

This creates a clear, consistent, and powerful user experience.

---

## STEP 3: IMPLEMENTATION DETAILS

### SafeTensor DisTorch (`override_class_with_distorch_safetensor`)
-   **Logic**: Uses the `virtual_ram_gb` to determine how many model blocks to offload to the selected `donor_device`.
-   **Block Discovery**: Employs a robust, multi-stage discovery mechanism to find swappable blocks in various model architectures (UNet, FLUX, etc.).
-   **Logging**: Generates high-quality, formatted tables detailing the block swap setup, identical in style to the GGUF logs.

### GGUF DisTorch (`override_class_with_distorch_gguf`)
-   **Logic**: A new wrapper has been created with the standardized UI. It translates the user's `virtual_ram_gb` and `donor_device` selection into the necessary allocation string for the GGUF backend.
-   **Backward Compatibility**: The previous GGUF wrapper has been renamed to `override_class_with_distorch_gguf_legacy` and moved to the `multigpu/legacy` category. This ensures that existing workflows will **not** break.

---

## STEP 4: CURRENT IMPLEMENTATION CODE IN __init__.py

The following code is a representation of the current implementation within `__init__.py`.

```python
# New Standardized GGUF Wrapper
def override_class_with_distorch_gguf(cls):
    """Standardized DisTorch wrapper for GGUF models."""
    # ... (Full implementation in __init__.py)

# Legacy GGUF Wrapper for Backward Compatibility
def override_class_with_distorch_gguf_legacy(cls):
    """Legacy DisTorch wrapper for GGUF models for backward compatibility."""
    # ... (Full implementation in __init__.py)

# Standardized SafeTensor Wrapper
def override_class_with_distorch_safetensor(cls):
    """DisTorch wrapper for SafeTensor models, providing block-swap memory optimization."""
    # ... (Full implementation in __init__.py)
```

---

## STEP 5: REGISTRATION IN __init__.py

The new standardized wrappers are registered, and the legacy GGUF nodes are preserved.

```python
# Register the new Standardized DisTorch GGUF wrappers
NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorchMultiGPU"] = override_class_with_distorch_gguf(UnetLoaderGGUF)
NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorchLegacyMultiGPU"] = override_class_with_distorch_gguf_legacy(UnetLoaderGGUF)

# Register the new Standardized DisTorch SafeTensor wrappers
NODE_CLASS_MAPPINGS["CheckpointLoaderSimpleDisTorchMultiGPU"] = override_class_with_distorch_safetensor(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
# ... and so on for all other core loaders.
```

---

## STEP 6: IMPLEMENTATION CHECKLIST

- [X] Standardize UI for both SafeTensor and GGUF DisTorch nodes (DONE)
- [X] Implement new GGUF wrapper with simplified controls (DONE)
- [X] Preserve old GGUF wrapper for backward compatibility under a "Legacy" name (DONE)
- [X] Update SafeTensor wrapper to match the new standardized UI (DONE)
- [X] Ensure SafeTensor block discovery is robust for models like FLUX (DONE)
- [X] Ensure SafeTensor logging matches the GGUF style (DONE)
- [ ] Test with SDXL checkpoint
- [ ] Test with Flux checkpoint
- [ ] Test with GGUF models (new and legacy nodes)
- [ ] Verify memory usage and performance

---

## FUTURE EXTENSIONS

1.  **Auto-mode**: Automatically determine optimal settings based on available VRAM and model size for both GGUF and SafeTensor.
2.  **Unified Expert Mode**: Explore a unified syntax for the `expert_mode_allocations` string that could apply to both model types.
