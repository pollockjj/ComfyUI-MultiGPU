# CLIPLoaderMultiGPU

The `CLIPLoaderMultiGPU` node is used to load CLIP text encoder models with device selection capability, enabling users to specify which GPU or device should be used for model execution.

This node automatically detects models located in the `ComfyUI/models/clip` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `clip_name` | `STRING` | The name of the CLIP model to load. |
| `type` | `STRING` | The type of CLIP model (e.g., 'stable_diffusion', 'stable_diffusion_xl'). |
| `device` | `STRING` | Target device for text encoder compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `CLIP` | `CLIP` | The loaded CLIP text encoder model. |
