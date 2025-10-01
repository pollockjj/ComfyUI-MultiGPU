# DiffControlNetLoaderMultiGPU

The `DiffControlNetLoaderMultiGPU` node is used to load Diffusers ControlNet models (HuggingFace Hub repositories) with device selection capability, enabling users to specify which GPU or device should be used for model execution.

This node loads ControlNet models directly from HuggingFace model repositories by specifying the repository ID (e.g., "diffusers/controlnet-canny-sdxl-1.0").

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_path` | `STRING` | The HuggingFace repository ID or local path of the diffusers ControlNet model to load. |
| `device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `CONTROL_NET` | `CONTROL_NET` | The loaded diffusers ControlNet model. |
