# LoadFluxControlNetMultiGPU

`LoadFluxControlNetMultiGPU` exposes device selection for XLabAI's FLUX ControlNet loader, letting you keep the ControlNet on a secondary GPU or the CPU while the main FLUX UNet stays on your primary compute device.

## Inputs

All inputs from the upstream `LoadFluxControlNet` node remain unchanged. The MultiGPU variant introduces one optional field:

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | MultiGPU device that will host the ControlNet during inference. |

## Outputs

Outputs match the base FLUX ControlNet loader exactly; only the device placement differs.
