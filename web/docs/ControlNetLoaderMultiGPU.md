# ControlNetLoaderMultiGPU

The `ControlNetLoaderMultiGPU` node is used to load ControlNet models with device selection capability, enabling users to specify which GPU or device should be used for model execution.

This node automatically detects models located in the `ComfyUI/models/controlnet` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `control_net_name` | `STRING` | The name of the ControlNet model to load. |
| `device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `CONTROL_NET` | `CONTROL_NET` | The loaded ControlNet model. |
