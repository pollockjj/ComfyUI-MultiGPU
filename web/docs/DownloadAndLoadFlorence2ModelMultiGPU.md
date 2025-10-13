# DownloadAndLoadFlorence2ModelMultiGPU

`DownloadAndLoadFlorence2ModelMultiGPU` mirrors the download-and-load helper supplied by `ComfyUI-Florence2`, but with explicit device and offload selection so large Florence2 checkpoints can live on secondary GPUs or CPU memory.

## Inputs

All original inputs from `DownloadAndLoadFlorence2Model` remain available. The MultiGPU wrapper introduces two optional selectors:

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Compute device to host the model once loaded. |
| `offload_device` | `STRING` | Device that receives automatic offloads (defaults to `cpu`). |

## Outputs

Outputs match the base Florence2 helper (model handle plus aux data). The only difference is that the returned model is already resident on the device you specified.
