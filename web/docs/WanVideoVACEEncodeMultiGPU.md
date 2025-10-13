# WanVideoVACEEncodeMultiGPU

`WanVideoVACEEncodeMultiGPU` encodes VACE reference inputs for WanVideo workflows while respecting your chosen device. It patches the base encoder to run on the MultiGPU device and to reuse Wan offload settings automatically.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `vae` | `WANVAE` | VAE pair to use during encoding. |
| `load_device` | `MULTIGPUDEVICE` | Device that will execute encoding. |
| `width` | `INT` | Target latent width. |
| `height` | `INT` | Target latent height. |
| `num_frames` | `INT` | Number of frames to encode. |
| `strength` | `FLOAT` | Overall conditioning strength. |
| `vace_start_percent` | `FLOAT` | Step fraction where VACE influence begins. |
| `vace_end_percent` | `FLOAT` | Step fraction where VACE influence ends. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `input_frames` | `IMAGE` | Input frames used for conditioning. |
| `ref_images` | `IMAGE` | Reference imagery to encode. |
| `input_masks` | `MASK` | Masks applied during encoding. |
| `prev_vace_embeds` | `WANVIDIMAGE_EMBEDS` | Prior VACE embeds to reuse or blend. |
| `tiled_vae` | `BOOLEAN` | Enable tiled encode for lower VRAM usage. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `vace_embeds` | `WANVIDIMAGE_EMBEDS` | Encoded VACE embeddings for Wan samplers. |
