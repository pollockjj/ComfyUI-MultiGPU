# WanVideoImageToVideoEncodeMultiGPU

`WanVideoImageToVideoEncodeMultiGPU` mirrors WanVideo's image-to-video encoder but ensures the heavy transformer work runs on your selected device while pushing temporary buffers to the configured offload target. Use it to convert reference imagery into Wan latents for I2V workflows.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `width` | `INT` | Output latent width (multiple of 8). |
| `height` | `INT` | Output latent height (multiple of 8). |
| `num_frames` | `INT` | Number of frames to encode. |
| `noise_aug_strength` | `FLOAT` | Noise level to add before encoding (helps motion). |
| `start_latent_strength` | `FLOAT` | Multiplier applied at sequence start. |
| `end_latent_strength` | `FLOAT` | Multiplier applied at sequence end. |
| `force_offload` | `BOOLEAN` | Offload Wan model once encoding finishes. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `vae` | `WANVAE` | VAE pair from Wan VAE loader; defaults to global VAE if omitted. |
| `load_device` | `MULTIGPUDEVICE` | Device to run encoding on. |
| `clip_embeds` | `WANVIDIMAGE_CLIPEMBEDS` | Additional clip guidance tensors. |
| `start_image` | `IMAGE` | First frame reference. |
| `end_image` | `IMAGE` | End frame reference for interpolation. |
| `control_embeds` | `WANVIDIMAGE_EMBEDS` | Control signal tensors (e.g., Fun). |
| `fun_or_fl2v_model` | `BOOLEAN` | Enable special behaviour for FLF2V/Fun models. |
| `temporal_mask` | `MASK` | Mask for temporal control. |
| `extra_latents` | `LATENT` | Additional latents to prepend (e.g., Skyreels refs). |
| `tiled_vae` | `BOOLEAN` | Use tiled VAE encoding to minimise VRAM. |
| `add_cond_latents` | `ADD_COND_LATENTS` | Extra conditional latents for advanced workflows. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `image_embeds` | `WANVIDIMAGE_EMBEDS` | Encoded Wan latents for downstream samplers. |
