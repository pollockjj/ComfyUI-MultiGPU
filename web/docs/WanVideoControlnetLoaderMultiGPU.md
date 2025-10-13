# WanVideoControlnetLoaderMultiGPU

`WanVideoControlnetLoaderMultiGPU` loads WanVideo-compatible ControlNets while letting you choose the execution device and optional FP8 quantisation modes.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `STRING` | ControlNet file from `ComfyUI/models/controlnet`. |
| `base_precision` | `STRING` | Weight precision (`fp32`, `bf16`, `fp16`). |
| `quantization` | `STRING` | FP8 preset (`disabled`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2`, `fp8_e4m3fn_fast_no_ffn`). |
| `load_device` | `STRING` | Wan loader slot (`main_device` or `offload_device`). |
| `device` | `STRING` | MultiGPU device that will host the ControlNet. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `controlnet` | `WANVIDEOCONTROLNET` | Loaded ControlNet ready for Wan samplers. |
