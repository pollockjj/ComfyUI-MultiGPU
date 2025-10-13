# PulidEvaClipLoaderMultiGPU

`PulidEvaClipLoaderMultiGPU` prepares the EVA CLIP text encoder required by PuLID and keeps it on the device you nominate for downstream conditioning.

## Inputs

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Device selected for the EVA CLIP encoder. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `eva_clip` | `EVA_CLIP` | Loaded EVA CLIP encoder instance. |
