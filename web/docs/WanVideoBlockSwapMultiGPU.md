# WanVideoBlockSwapMultiGPU

`WanVideoBlockSwapMultiGPU` prepares block swap arguments for WanVideo models and adds an explicit `swap_device` selector so you can decide which device receives swapped transformer blocks.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| *(base Wan block swap inputs)* | *varies* | All parameters exposed by the upstream `WanVideoBlockSwap` node are available and behave identically. |
| `swap_device` | `STRING` | Additional MultiGPU device option that picks the destination for swapped layers (`cpu`, `cuda:1`, etc.). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `block_swap_args` | `BLOCKSWAPARGS` | Configuration dictionary to feed into `WanVideoModelLoaderMultiGPU` or Wan samplers. |
