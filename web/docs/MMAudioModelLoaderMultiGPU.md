# MMAudioModelLoaderMultiGPU

`MMAudioModelLoaderMultiGPU` loads MMAudio diffusion checkpoints while letting you pin the model weights to a specific compute device. Use it to keep long-running audio generations off your primary image GPU.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `mmaudio_model` | `STRING` | Model filename from `ComfyUI/models/mmaudio`. |
| `base_precision` | `STRING` | Weight precision to request (`fp16`, `fp32`, `bf16`). |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Target device for the loaded model (e.g. `cuda:0`, `cpu`). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `mmaudio_model` | `MMAUDIO_MODEL` | Loaded MMAudio diffusion pipeline. |
