# UnetLoaderGGUFAdvancedMultiGPU

The `UnetLoaderGGUFAdvancedMultiGPU` node is used to load GGUF format UNet models with device selection capability, enabling users to specify which GPU or device should be used for model execution.

This node automatically detects models located in the `ComfyUI/models/unet_gguf` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `unet_name` | `STRING` | The name of the GGUF format UNet model to load. |
| `dequant_dtype` | `STRING` | Target data type for model dequantization during loading (options: 'default', 'target', 'float32', 'float16', 'bfloat16'). |
| `patch_dtype` | `STRING` | Data type for LoRA patches applied to the model (options: 'default', 'target', 'float32', 'float16', 'bfloat16'). |
| `patch_on_device` | `BOOLEAN` | Whether to apply LoRA patches directly on the target device. |
| `device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded GGUF format UNet model with advanced quantization settings. |
