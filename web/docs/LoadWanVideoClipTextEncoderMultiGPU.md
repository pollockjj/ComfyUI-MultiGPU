# LoadWanVideoClipTextEncoderMultiGPU

`LoadWanVideoClipTextEncoderMultiGPU` loads WanVideo CLIP vision/text encoders on the device you specify, making it easy to keep encoders off your primary compute GPU when memory is tight.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_name` | `STRING` | CLIP vision or text encoder model from `ComfyUI/models/clip_vision` or `ComfyUI/models/text_encoders`. |
| `precision` | `STRING` | Weight precision for the model (`fp16`, `fp32`, or `bf16`). |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Target MultiGPU device to host the encoder. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `wan_clip_vision` | `CLIP_VISION` | Loaded CLIP vision/text module ready for image conditioning. |
| `load_device` | `MULTIGPUDEVICE` | Device that now owns the encoder; feed into `WanVideoClipVisionEncode`. |
