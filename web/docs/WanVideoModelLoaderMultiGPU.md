# WanVideoModelLoaderMultiGPU

`WanVideoModelLoaderMultiGPU` wraps the base WanVideo model loader so you can pick both the loader device and the downstream compute device when working with large WanVideo checkpoints. The node patches the underlying WanVideo loader so the model materialises on the device chosen via `compute_device` while still honouring WanVideo's block swap and quantisation options.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `STRING` | Model file from `ComfyUI/models/diffusion_models` or `ComfyUI/models/unet_gguf` to load. |
| `base_precision` | `STRING` | Floating-point format for base weights (`fp32`, `bf16`, `fp16`, or `fp16_fast`). |
| `quantization` | `STRING` | Optional FP8 quantisation preset; `disabled` keeps original precision. |
| `load_device` | `STRING` | WanVideo loader slot (`main_device` or `offload_device`) used during initial weight materialisation. |
| `compute_device` | `STRING` | MultiGPU device id (e.g. `cuda:0`, `cuda:1`, `cpu`) to run inference on. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `attention_mode` | `STRING` | Select specialised attention kernels (`sdpa`, `flash_attn_2`, `flash_attn_3`, `sageattn`, `sageattn_3`, `radial_sage_attention`). |
| `compile_args` | `WANCOMPILEARGS` | Torch compile configuration passed through to WanVideo. |
| `block_swap_args` | `BLOCKSWAPARGS` | Enables WanVideo block swapping; supply alongside `WanVideoBlockSwapMultiGPU`. |
| `lora` | `WANVIDLORA` | Optional Wan LoRA bundle to apply during load. |
| `vram_management_args` | `VRAM_MANAGEMENTARGS` | DiffSynth-Studio memory manager arguments for aggressive VRAM reclamation. |
| `extra_model` | `VACEPATH` | Adds auxiliary model weights (e.g. VACE / MTV Crafter). |
| `fantasytalking_model` | `FANTASYTALKINGMODEL` | Preloads FantasyTalking speech model. |
| `multitalk_model` | `MULTITALKMODEL` | Preloads MultiTalk model. |
| `fantasyportrait_model` | `FANTASYPORTRAITMODEL` | Preloads FantasyPortrait model. |
| `rms_norm_function` | `STRING` | Choose RMSNorm implementation (`default` Wan variant or `pytorch`). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `model` | `WANVIDEOMODEL` | Initialised Wan diffusion model ready for sampling. |
| `compute_device` | `MULTIGPUDEVICE` | Device id chosen for downstream nodes; pass directly into samplers. |
