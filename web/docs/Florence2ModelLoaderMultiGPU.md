# Florence2ModelLoaderMultiGPU

`Florence2ModelLoaderMultiGPU` wraps the Florence2 model loader so you can decide which device handles model inference and which device receives Wan/Comfy offloads. Use it exactly like the original node from `ComfyUI-Florence2`; all native inputs remain available.

## Inputs

All parameters from `Florence2ModelLoader` are still supported. The MultiGPU variant adds the following optional fields:

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | MultiGPU device used for runtime compute (`cuda:0`, `cuda:1`, `cpu`, etc.). |
| `offload_device` | `STRING` | Device that receives automatic model offloads (defaults to `cpu`). |

## Outputs

The outputs are identical to the upstream Florence2 loader (model tuple, additional metadata). Use them interchangeably in existing workflows; only the device placement behaviour changes.
