# LoadWanVideoT5TextEncoderMultiGPU

`LoadWanVideoT5TextEncoderMultiGPU` loads WanVideo T5 text encoders while letting you choose the MultiGPU device used for embedding work. The node returns both the encoder handle and the device string so downstream text nodes inherit placement automatically.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_name` | `STRING` | T5 model from `ComfyUI/models/text_encoders`. |
| `precision` | `STRING` | Base precision for the encoder (`fp32` or `bf16`). |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | MultiGPU device (defaults to secondary GPU when available). |
| `quantization` | `STRING` | Enable FP8 quantisation (`fp8_e4m3fn`) when supported. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `wan_t5_model` | `WANTEXTENCODER` | Loaded Wan T5 encoder bundle. |
| `load_device` | `MULTIGPUDEVICE` | Device string to reuse with `WanVideoTextEncode*` nodes. |
