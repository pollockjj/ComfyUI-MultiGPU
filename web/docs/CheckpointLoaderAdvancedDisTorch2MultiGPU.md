# CheckpointLoaderAdvancedDisTorch2MultiGPU

The `CheckpointLoaderAdvancedDisTorch2MultiGPU` node is used to load checkpoint models with advanced DisTorch2 distributed tensor allocation, providing granular control over UNet, CLIP, and VAE component allocation across multiple devices with independent virtual VRAM management.

This node automatically detects models located in the `ComfyUI/models/checkpoints` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `ckpt_name` | `STRING` | The name of the checkpoint model to load. |
| `unet_compute_device` | `STRING` | Target compute device for UNet distributed allocation (e.g., 'cuda:0', 'cuda:1', 'cpu'). |
| `unet_virtual_vram_gb` | `FLOAT` | Amount of virtual VRAM in gigabytes for UNet component distributed allocation (default: 4.0, range: 0.0-128.0). |
| `unet_donor_device` | `STRING` | Device to donate VRAM from when allocating UNet virtual memory (default: 'cpu'). |
| `clip_compute_device` | `STRING` | Target compute device for CLIP distributed allocation (default: 'cpu'). |
| `clip_virtual_vram_gb` | `FLOAT` | Amount of virtual VRAM in gigabytes for CLIP component distributed allocation (default: 2.0, range: 0.0-128.0). |
| `clip_donor_device` | `STRING` | Device to donate VRAM from when allocating CLIP virtual memory (default: 'cpu'). |
| `vae_device` | `STRING` | Target device for the VAE component (e.g., 'cuda:0', 'cuda:1', 'cpu'). |
| `unet_expert_mode_allocations` | `STRING` | Advanced UNet allocation string for expert device/ratio distributions. |
| `clip_expert_mode_allocations` | `STRING` | Advanced CLIP allocation string for expert device/ratio distributions. |
| `high_precision_loras` | `BOOLEAN` | Whether to use high-precision LoRA patches (default: true). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded UNet diffusion model with DisTorch2 distributed allocation. |
| `CLIP` | `CLIP` | The loaded CLIP text encoder model with DisTorch2 distributed allocation. |
| `VAE` | `VAE` | The loaded VAE decoder/encoder model. |

## DisTorch2 Distributed Loading

DisTorch2 is an advanced memory management system that enables loading and running large diffusion models across multiple GPUs by intelligently distributing tensor allocations. Instead of loading an entire model on a single device, DisTorch2 splits the model's layers across available devices while maintaining computational efficiency.

This advanced checkpoint loader provides independent DisTorch2 allocation control for UNet and CLIP components, while using standard device placement for VAE.

### Key Concepts

**Individual Component Control**: Each model component (UNet, CLIP, VAE) can have its own allocation strategy.

**Virtual VRAM Allocation**: Artificially increases the available VRAM on compute devices by borrowing memory capacity from donor devices through intelligent tensor distribution.

**Expert Mode Allocations**: Advanced users can manually specify exactly how much of each component should be placed on each device using ratio or byte-based allocation strings.

### Allocation Examples

**Basic Virtual VRAM Mode**:
- `unet_compute_device`: `cuda:0`, `unet_virtual_vram_gb`: `8.0`, `unet_donor_device`: `cuda:1`
- `clip_compute_device`: `cuda:1`, `clip_virtual_vram_gb`: `2.0`, `clip_donor_device`: `cpu`
- Result: UNet loads as if cuda:0 has 8GB more VRAM, CLIP loads with cuda:1 having 2GB more capacity.

**Expert Ratio Allocation**:
- `unet_expert_mode_allocations`: `cuda:0,70%;cuda:1,30%`
- `clip_expert_mode_allocations`: `cuda:1,50%;cpu,50%`
- Distributes UNet with 70% on GPU 0, 30% on GPU 1, and CLIP with 50% on GPU 1, 50% on CPU.
