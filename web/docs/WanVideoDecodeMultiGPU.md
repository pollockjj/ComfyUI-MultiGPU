# WanVideoDecodeMultiGPU

`WanVideoDecodeMultiGPU` decodes Wan latents back into frames using the VAE you provide, pinning decode work to the chosen MultiGPU device and safeguarding validation for tiled decode settings.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `vae` | `WANVAE` | VAE pair from a Wan VAE loader. |
| `load_device` | `MULTIGPUDEVICE` | Device that will run the decode. |
| `samples` | `LATENT` | Latent tensor to decode. |
| `enable_vae_tiling` | `BOOLEAN` | Enables tiled decoding to reduce VRAM usage. |
| `tile_x` | `INT` | Tile width in pixels. |
| `tile_y` | `INT` | Tile height in pixels. |
| `tile_stride_x` | `INT` | Horizontal stride between tiles. |
| `tile_stride_y` | `INT` | Vertical stride between tiles. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `normalization` | `STRING` | Switch between default and min-max output normalisation. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `images` | `IMAGE` | Decoded video frames or image batch. |
