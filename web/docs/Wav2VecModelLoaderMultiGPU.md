# Wav2VecModelLoaderMultiGPU

`Wav2VecModelLoaderMultiGPU` loads locally stored Wav2Vec2 models for WanVideo workflows while exposing MultiGPU placement controls and load-device selection.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `STRING` | Wav2Vec2 model from `ComfyUI/models/wav2vec2`. |
| `base_precision` | `STRING` | Weight precision (`fp32`, `bf16`, `fp16`). |
| `load_device` | `STRING` | Wan loader slot (`main_device` or `offload_device`). |
| `device` | `STRING` | MultiGPU device to own the model during inference. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `wav2vec_model` | `WAV2VECMODEL` | Loaded speech model for Wan audio pipelines. |
