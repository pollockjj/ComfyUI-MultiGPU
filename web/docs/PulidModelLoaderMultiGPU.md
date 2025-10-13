# PulidModelLoaderMultiGPU

`PulidModelLoaderMultiGPU` loads PuLID identity preservation checkpoints onto your chosen device so facial guidance workloads can avoid your primary rendering GPU.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `pulid_file` | `STRING` | PuLID model file from `ComfyUI/models/pulid`. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Device that will host the PuLID weights. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `model` | `PULID` | Loaded PuLID model bundle. |
