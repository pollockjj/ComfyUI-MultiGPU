# DiffusersLoaderMultiGPU

The `DiffusersLoaderMultiGPU` node is used to load Diffusers models (HuggingFace Hub repositories) with device selection capability, enabling users to specify which GPU or device should be used for model execution.

This node loads models directly from HuggingFace model repositories by specifying the repository ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0").

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_path` | `STRING` | The HuggingFace repository ID or local path of the diffusers model to load (e.g., 'stabilityai/stable-diffusion-xl-base-1.0'). |
| `device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded diffusers model. |
