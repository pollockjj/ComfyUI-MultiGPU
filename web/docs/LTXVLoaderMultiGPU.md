# LTXVLoaderMultiGPU

`LTXVLoaderMultiGPU` wraps `ComfyUI-LTXVideo`'s checkpoint loader so you can push LTX Video models to any GPU (or CPU) in your system without editing the base node.

## Inputs

Every input from the upstream `LTXVLoader` node is preserved. The MultiGPU version adds a single optional selector:

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | MultiGPU device that should host the loaded LTX Video checkpoint. |

## Outputs

Outputs are identical to the original LTX Video loader. The loader simply ensures the returned model already resides on the selected device.
