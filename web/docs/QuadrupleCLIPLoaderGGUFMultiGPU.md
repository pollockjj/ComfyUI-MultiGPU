# QuadrupleCLIPLoaderGGUFMultiGPU

The `QuadrupleCLIPLoaderGGUFMultiGPU` node is used to load quadruple GGUF format CLIP text encoder models with device selection capability, enabling users to specify which GPU or device should be used for model execution.

This node automatically detects models located in the `ComfyUI/models/clip` and `ComfyUI/models/clip_gguf` folders, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `clip_name1` | `STRING` | The name of the first CLIP model to load from combined clip and clip_gguf folders. |
| `clip_name2` | `STRING` | The name of the second CLIP model to load from combined clip and clip_gguf folders. |
| `clip_name3` | `STRING` | The name of the third CLIP model to load from combined clip and clip_gguf folders. |
| `clip_name4` | `STRING` | The name of the fourth CLIP model to load from combined clip and clip_gguf folders. |
| `device` | `STRING` | Target device for text encoder compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `CLIP` | `CLIP` | The loaded quadruple CLIP text encoder models. |
