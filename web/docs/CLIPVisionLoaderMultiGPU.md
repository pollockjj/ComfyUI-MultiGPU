# CLIPVisionLoaderMultiGPU

The `CLIPVisionLoaderMultiGPU` node is used to load CLIP Vision models with device selection capability, enabling users to specify which GPU or device should be used for vision encoder execution.

This node automatically detects models located in the `ComfyUI/models/clip_vision` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `clip_vision` | `STRING` | The name of the CLIP Vision model to load. |
| `device` | `STRING` | Target device for vision encoder compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `CLIP_VISION` | `CLIP_VISION` | The loaded CLIP Vision model. |
