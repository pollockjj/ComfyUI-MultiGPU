# CheckpointLoaderAdvancedMultiGPU

The `CheckpointLoaderAdvancedMultiGPU` node is used to load checkpoint models (complete diffusion models containing UNet, CLIP, and VAE components) with granular device control, allowing individual placement of each model component on different GPUs or devices.

This node automatically detects models located in the `ComfyUI/models/checkpoints` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `ckpt_name` | `STRING` | The name of the checkpoint model to load. |
| `unet_device` | `STRING` | Target device for the UNet diffusion model component (e.g., 'cuda:0', 'cuda:1', 'cpu'). |
| `clip_device` | `STRING` | Target device for the CLIP text encoder component (e.g., 'cuda:0', 'cuda:1', 'cpu'). |
| `vae_device` | `STRING` | Target device for the VAE decoder/encoder component (e.g., 'cuda:0', 'cuda:1', 'cpu'). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded UNet diffusion model. |
| `CLIP` | `CLIP` | The loaded CLIP text encoder model. |
| `VAE` | `VAE` | The loaded VAE decoder/encoder model. |
