# WanVideoVAELoaderMultiGPU

`WanVideoVAELoaderMultiGPU` loads WanVideo VAEs on the device you choose, returning both the VAE handle and the selected device so downstream encode/decode nodes run on the correct hardware.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_name` | `STRING` | VAE model from `ComfyUI/models/vae`. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `load_device` | `STRING` | Destination MultiGPU device. |
| `precision` | `STRING` | VAE precision (`fp16`, `fp32`, or `bf16`). |
| `compile_args` | `WANCOMPILEARGS` | Optional torch compile parameters. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `vae` | `WANVAE` | Loaded Wan VAE model. |
| `load_device` | `MULTIGPUDEVICE` | Device string to feed into encode/decode nodes. |
