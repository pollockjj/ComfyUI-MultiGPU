# FantasyTalkingModelLoaderMultiGPU

`FantasyTalkingModelLoaderMultiGPU` loads FantasyTalking diffusion models with explicit device control, making it easier to keep speech animation workloads off your primary compute GPU.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `STRING` | FantasyTalking model from `ComfyUI/models/diffusion_models`. |
| `base_precision` | `STRING` | Precision for the weights (`fp32`, `bf16`, `fp16`). |
| `device` | `STRING` | MultiGPU device that should host the model. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `model` | `FANTASYTALKINGMODEL` | Loaded FantasyTalking model bundle. |
