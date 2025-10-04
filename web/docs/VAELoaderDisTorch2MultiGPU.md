# VAELoaderDisTorch2MultiGPU

The `VAELoaderDisTorch2MultiGPU` node is used to load VAE (Variational Autoencoder) models with DisTorch2 distributed tensor allocation, enabling advanced multi-device VRAM management to handle larger models across multiple GPUs.

This node automatically detects models located in the `ComfyUI/models/vae` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `vae_name` | `STRING` | The name of the VAE model to load. |
| `compute_device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |
| `virtual_vram_gb` | `FLOAT` | Amount of virtual VRAM in gigabytes to allocate for distributed tensor management (default: 4.0, range: 0.0-128.0). |
| `donor_device` | `STRING` | Device to donate VRAM from when allocating virtual memory (default: 'cpu'). |
| `expert_mode_allocations` | `STRING` | Advanced allocation string for expert users to manually specify device/ratio distributions (e.g., 'cuda:0,50%;cpu,*'). |
| `eject_models` | `BOOLEAN` | Whether to unload ALL models from the target device before loading this model, enabling deterministic model eviction for testing and memory management (default: true). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `VAE` | `VAE` | The loaded VAE decoder/encoder with DisTorch2 distributed allocation applied. |

## DisTorch2 Distributed Loading

DisTorch2 is an advanced memory management system that enables loading and running large diffusion models across multiple GPUs by intelligently distributing tensor allocations. Instead of loading an entire model on a single device, DisTorch2 splits the model's layers across available devices while maintaining computational efficiency.

### Key Concepts

**Virtual VRAM Allocation**: Artificially increases the available VRAM on the compute device by borrowing memory capacity from donor devices through intelligent tensor distribution.

**Expert Mode Allocations**: Advanced users can manually specify exactly how much of the model should be placed on each device using ratio or byte-based allocation strings.

### Allocation Examples

**Basic Virtual VRAM Mode**:
- `compute_device`: `cuda:0`
- `virtual_vram_gb`: `8.0`
- `donor_device`: `cuda:1`
- Result: Loads model as if cuda:0 had 8GB more VRAM available, using cuda:1 as memory donor.

**Expert Ratio Allocation**:
- `expert_mode_allocations`: `cuda:0,60%;cuda:1,30%;cpu,10%`
- Distributes model layers with 60% on GPU 0, 30% on GPU 1, and 10% on CPU.

**Expert Byte Allocation**:
- `expert_mode_allocations`: `cuda:0,4gb;cuda:1,2gb;cpu,*`
- Allocates exactly 4GB to cuda:0, 2GB to cuda:1, and remaining to CPU.

**Mixed Mode**:
Combines virtual VRAM with expert allocations for complex multi-device scenarios.
