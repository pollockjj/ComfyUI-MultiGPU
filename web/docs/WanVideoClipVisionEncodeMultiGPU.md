# WanVideoClipVisionEncodeMultiGPU

`WanVideoClipVisionEncodeMultiGPU` runs WanVideo's CLIP vision encoder on the device you provide. It supports tiled encoding, dual-image blending, and optional negative guidance while managing offload behaviour for you.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `clip_vision` | `CLIP_VISION` | Encoder pair from `LoadWanVideoClipTextEncoderMultiGPU`. |
| `load_device` | `MULTIGPUDEVICE` | Device where encoding should occur. |
| `image_1` | `IMAGE` | Primary image to encode. |
| `strength_1` | `FLOAT` | Weight applied to the first image embedding. |
| `strength_2` | `FLOAT` | Weight applied to the second image embedding. |
| `crop` | `STRING` | Cropping mode (`center` or `disabled`). |
| `combine_embeds` | `STRING` | Strategy when combining multiple embeds (`average`, `sum`, `concat`, `batch`). |
| `force_offload` | `BOOLEAN` | Offload encoder after processing. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `image_2` | `IMAGE` | Secondary image for combination. |
| `negative_image` | `IMAGE` | Negative reference image. |
| `tiles` | `INT` | Enable Matteo's tiled encode by setting tile count > 0. |
| `ratio` | `FLOAT` | Blend ratio used with tiled encoding. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `image_embeds` | `WANVIDIMAGE_CLIPEMBEDS` | CLIP vision embeddings suitable for Wan samplers or encoders. |
