# ComfyUI-MultiGPU Tensor Identity Implementation

## Current Issue
When tensors move between devices with `.to()`, their identity changes and hash value changes with it. This makes it impossible to track tensors through their lifecycle, breaking patch application and caching mechanisms.

## Implementation Plan

1. Add hash preservation function
   ```python
   def register_patched_ggml_tensor_hash():
       # Add original_hash attribute to GGMLTensor
       # Patch to() method to preserve original_hash during transfers
       # Add update_hash() method for explicit hash refreshing
   ```

2. Get hash data after first to() operation
   ```python
   # In model_patcher.load after to() is called:
   original_hash = getattr(parameter_value, "original_hash", None)
   if original_hash:
       # Store in cached_tensor_map for tracking
   ```

3. Print all aspects of the tensor including original_hash
   ```python
   # Add to logging string:
   f"... | original_hash=0x{original_hash:x} | ..."
   ```

## Implementation Locations
- `register_patched_ggml_tensor_hash()` in __init__.py (not being called early enough)
- Hash capture after module_object.to() in patched model_patcher.load
- Hash display in standard tensor info logging string