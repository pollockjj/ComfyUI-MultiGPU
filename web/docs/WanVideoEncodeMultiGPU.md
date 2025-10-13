# WanVideoEncodeMultiGPU

`WanVideoEncodeMultiGPU` encodes single images into Wan latents using the selected device, mirroring WanVideo's image encoder while adding explicit device routing and tiled encode safeguards.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `vae` | `WANVAE` | VAE pair for encoding. |
| `load_device` | `MULTIGPUDEVICE` | Device that will run the encode. |
| `image` | `IMAGE` | Image tensor to convert to latents. |
| `enable_vae_tiling` | `BOOLEAN` | Enables tiled encoding to lower VRAM usage. |
| `tile_x` | `INT` | Tile width in pixels. |
| `tile_y` | `INT` | Tile height in pixels. |
| `tile_stride_x` | `INT` | Horizontal stride between tiles. |
| `tile_stride_y` | `INT` | Vertical stride between tiles. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `noise_aug_strength` | `FLOAT` | Adds noise before encoding for motion workflows. |
| `latent_strength` | `FLOAT` | Scales encoded latents. |
| `mask` | `MASK` | Optional mask to limit encoding region. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `samples` | `LATENT` | Encoded Wan latent tensor. |
