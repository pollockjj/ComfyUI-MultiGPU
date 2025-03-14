# ComfyUI-MultiGPU Tensor Identity Implementation

## Current Issue
When tensors move between devices with `.to()`, their identity changes and hash value changes with it. This makes it impossible to track tensors through their lifecycle, breaking patch application and caching mechanisms.

## Implementation Status

The GGMLTensor class in ComfyUI-GGUF/ops.py now sets original_hash at initialization and preserves it during device transfers:

```python
def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
    super().__init__()
    self.tensor_type = tensor_type
    self.tensor_shape = tensor_shape
    self.patches = patches
    self.original_hash = self.__hash__()

def to(self, *args, **kwargs):
    new = super().to(*args, **kwargs)
    new.tensor_type = getattr(self, "tensor_type", None)
    new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
    new.patches = getattr(self, "patches", []).copy()
    new.original_hash = getattr(self, "original_hash", None)
    return new
```

The to() method now safely handles the case where original_hash might not exist by using getattr with a default value.

## Latest Update (March 2025)

We've successfully implemented a full lifecycle tracking system for tensors:

1. In ComfyUI-GGUF/ops.py:
   - GGMLTensor now sets original_hash at initialization
   - original_hash is preserved during device transfers via the to() method

2. In ComfyUI-MultiGPU/ggml_weight_utils.py:
   - cast_bias_weight_patched is modified to use original_hash for tracking tensors
   - get_weight function creates and maintains cached_tensor_map
   - Debug print statements confirm tensors maintain identity across device transfers

3. In ComfyUI-MultiGPU/__init__.py:
   - GGMLLayer.get_weight is patched to use our enhanced version
   - Added storage in distorch_load_map to track tensor assignments

We can now track tensors throughout their entire lifecycle, from initialization through all device transfers, addressing the core issue of maintaining tensor identity. Our detailed debug output confirms tensors maintain their identity even when moved between devices:

```
TENSOR CACHE: ptr=0x747e9e72f290 | index=185 | name=diffusion_model.double_blocks.18.img_mlp.0.weight | patches=6 | device=cpu | size=38.25MB
```

## Code Structure and Expectations

This is DETERMINISTIC, HIGH-PERFORMANCE inference code with the following characteristics:

1. All operations MUST be fully deterministic - no exception handling for edge cases that should never occur
2. Memory management is explicit and precise - objects that are None or incorrectly typed indicate bugs, not expected states
3. The caching system expects consistent tensor tracking - any tensor hash change breaks the entire pipeline
4. Error handling through assertions/crashes is EXPECTED AND DESIRED for data integrity issues
5. Code uses hash-based identity tracking with these assumptions:
   - A tensor's original_hash is NEVER None once initialized
   - Tensor hash values are ALWAYS consistent across device transfers
   - Every tensor in cached_tensor_map has a valid original_hash as its key
   - No defensive coding for None checks - values should NEVER be None

6. The transitions from tensor-keyed maps to hash-keyed maps must maintain these invariants:
   - No None values as keys in maps
   - No "default values" that hide bugs
   - Hash values must be consistently tracked and preserved

## Critical Implementation Invariants

1. When using get_weight, source_tensor_hash MUST be provided and valid
2. All tensor tracking maps must use the same consistent key type (either tensor object or hash)
3. Debug prints track exact values to ensure consistency, not to handle errors
4. Cache initialization assumes valid keys (no None values) throughout the maps
5. When a None is encountered, it represents a BUG, not an expected state to handle

## Debug Findings
1. Setting original_hash at initialization ensures every GGMLTensor has a consistent hash from creation
2. The tensor's identity is preserved during device transfers via the .to() method
3. The original_hash attribute is used for tracking tensors in both cached_tensor_map and distorch_load_map correctly
4. Debug output shows tensors maintain their identity even when moved to different devices
5. Our improved diagnostic output helps identify all tensors through the inference process

## Implementation Locations
- Identity setting: GGMLTensor.__init__ in ComfyUI-GGUF/ops.py
- Identity preservation: GGMLTensor.to() in ComfyUI-GGUF/ops.py 
- Tensor tracking: get_weight and cast_bias_weight_patched in ComfyUI-MultiGPU/ggml_weight_utils.py
- Runtime patching: register_patched_gguf_get_weight in ComfyUI-MultiGPU/__init__.py

## Key Takeaways
1. Tensors now maintain identity from creation through all device transfers
2. The system can track tensors even when they're distributed across multiple devices
3. Debug output confirms tensor identity is preserved through the complete lifecycle
4. We can now build more advanced caching and prefetching systems with confidence that tensor identity tracking is reliable

## Planned Prefetch System
The get_weight function contains comments outlining a sophisticated prefetching system:

1. For tensors with cache_level="uninitialized":
   - Filter for tensors on CPU devices
   - Sort tensors by their inference index
   - Identify tensors N positions ahead of current index
   - Prefetch these tensors into a "ggml_prefetch_buffer" using non-blocking operations
   - Update cache_level to "ggml_prefetch"

2. For tensors being processed:
   - Check if the tensor exists in the prefetch buffer
   - If present, use the prefetched version (fast path)
   - If not, load normally from DRAM (slow path)

This deterministic prefetching system will significantly speed up inference by overlapping compute and memory operations, especially for CPU-bound tensors.