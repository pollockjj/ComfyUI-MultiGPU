# DownloadAndLoadWav2VecModelMultiGPU

`DownloadAndLoadWav2VecModelMultiGPU` downloads a preset Wav2Vec2 checkpoint from Hugging Face (if missing) and loads it onto the device you choose, mirroring WanVideo's helper while adding MultiGPU awareness.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `STRING` | Preset identifier (`TencentGameMate/chinese-wav2vec2-base` or `facebook/wav2vec2-base-960h`). |
| `base_precision` | `STRING` | Weight precision (`fp32`, `bf16`, `fp16`). |
| `load_device` | `STRING` | Wan loader slot (`main_device` or `offload_device`). |
| `device` | `STRING` | MultiGPU device to run the audio model. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `wav2vec_model` | `WAV2VECMODEL` | Downloaded and loaded Wav2Vec2 model. |
