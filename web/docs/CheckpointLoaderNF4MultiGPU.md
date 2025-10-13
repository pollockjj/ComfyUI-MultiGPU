# CheckpointLoaderNF4MultiGPU

`CheckpointLoaderNF4MultiGPU` wraps the NF4 checkpoint loader from `ComfyUI_bitsandbytes_NF4` so you can pick the execution device when working with 4-bit Quantised diffusion checkpoints.

## Inputs

All base parameters from `CheckpointLoaderNF4` are retained. The MultiGPU wrapper adds one optional field:

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Device that should own the loaded NF4 checkpoint (GPU id or `cpu`). |

## Outputs

Outputs are identical to the upstream NF4 loader (UNet/CLIP/VAE tuple). The only behavioural change is the explicit device placement. |
