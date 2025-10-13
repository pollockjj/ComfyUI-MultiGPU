# UNetLoaderLP

`UNetLoaderLP` is a low-precision variant of the standard UNet loader that disables high-precision LoRA tensors for CPU-stored models, freeing additional host memory while remaining compatible with MultiGPU device routing.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `unet_name` | `STRING` | UNet checkpoint filename from `ComfyUI/models/unet`. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Device that should serve the UNet after loading. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `model` | `MODEL` | Loaded UNet model with low-precision LoRA flag set. |
