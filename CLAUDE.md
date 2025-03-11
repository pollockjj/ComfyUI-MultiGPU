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

## Debug Findings
1. Setting original_hash at initialization ensures every GGMLTensor has a consistent hash from creation
2. The tensor's identity is preserved during device transfers via the .to() method
3. The original_hash attribute is used for tracking tensors in cached_tensor_map correctly
4. Debug output shows tensor names are correctly matched with their original_hash values
5. Our modified cast_bias_weight function in __init__.py confirms that tensor identity is maintained

## Implementation Locations
- Identity setting happens in GGMLTensor.__init__ in ComfyUI-GGUF/ops.py
- Identity preservation happens in GGMLTensor.to() in ComfyUI-GGUF/ops.py 
- Debug output in patched cast_bias_weight confirms tensors are correctly tracked by hash

## Dev Notes
The ggml_weight_utils_dev.py approach uses tensor pointers for lookup while the current approach uses original_hash. Both approaches have their merits, but using original_hash is more robust for tracking tensors across device moves.

## Key Takeaways
1. Tensors now maintain identity from creation through device transfers
2. The caching system correctly identifies tensors by their original_hash
3. Debug prints in cast_bias_weight confirm this identity is preserved throughout the lifecycle