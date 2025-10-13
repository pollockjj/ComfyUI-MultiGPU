# WanVideoTinyVAELoaderMultiGPU

`WanVideoTinyVAELoaderMultiGPU` loads lightweight Wan VAEs from the `vae_approx` folder, useful for preview drafts or efficiency workflows. The node mirrors ComfyUI's tiny VAE loader while exposing explicit device placement and optional parallel decoding.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_name` | `STRING` | Tiny VAE filename from `ComfyUI/models/vae_approx`. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `load_device` | `STRING` | MultiGPU device to host the VAE. |
| `precision` | `STRING` | Weight precision (`fp16`, `fp32`, `bf16`). |
| `parallel` | `BOOLEAN` | Enable parallel encode/decode for extra speed (uses more VRAM). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `vae` | `WANVAE` | Loaded lightweight VAE. |
| `load_device` | `MULTIGPUDEVICE` | Device string to pass into Wan encode/decode nodes. |
