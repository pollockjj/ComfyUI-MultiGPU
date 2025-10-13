# WanVideoUni3C_ControlnetLoaderMultiGPU

`WanVideoUni3C_ControlnetLoaderMultiGPU` loads Uni3C ControlNets for WanVideo, exposing device, attention, and compile options so you can balance performance and VRAM across multiple GPUs.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `STRING` | Uni3C ControlNet from `ComfyUI/models/controlnet`. |
| `base_precision` | `STRING` | Weight precision (`fp32`, `bf16`, `fp16`). |
| `load_device` | `STRING` | Wan loader slot (`main_device` or `offload_device`). |
| `device` | `STRING` | MultiGPU device that will host the ControlNet. |
| `quantization` | `STRING` | FP8 mode (`disabled`, `fp8_e4m3fn`, `fp8_e5m2`). |
| `attention_mode` | `STRING` | Attention kernel (`sdpa` or `sageattn`). |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `compile_args` | `WANCOMPILEARGS` | Torch compile configuration for the ControlNet. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `controlnet` | `WANVIDEOCONTROLNET` | Loaded Uni3C ControlNet ready for Wan samplers. |
