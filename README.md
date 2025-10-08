# ComfyUI-MultiGPU v2: Universal .safetensors and GGUF Multi-GPU Distribution with DisTorch
<p align="center">
  <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/assets/distorch_average.png" width="600">
  <br>
  <em>Free almost all of your GPU for what matters: Maximum latent space processing</em>
</p>

## The Core of ComfyUI-MultiGPU v2:
[^1]: This **enhances memory management,** not parallel processing. Workflow steps still execute sequentially, but with components (in full or in part) loaded across your specified devices. *Performance gains* come from avoiding repeated model loading/unloading when VRAM is constrained. *Capability gains* come from offloading as much of the model (VAE/CLIP/UNet) off of your main **compute** device as possible—allowing you to maximize latent space for actual computation.

1.  **Universal .safetensors Support**: Native DisTorch2 distribution for all `.safetensors` models.
2.  **Up to 10% Faster GGUF Inference versus DisTorch1**: The new DisTorch2 logic provides potential speedups for GGUF models versus the DisTorch V1 method.
3.  **Bespoke WanVideoWrapper Integration**: Tightly integrated, stable support for WanVideoWrapper with eight bespoke MultiGPU nodes.
4.  **New Model-Driven Allocation Options**: Two new inutuitive model-driven Expert Modes to facilitate exact placement on all available devices - 'bytes' and 'ratio'

<h1 align="center">DisTorch: How It Works</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/assets/distorch2_0.gif" width="800">
  <br>
  <em>DisTorch 2.0 in Action</em>
</p>

What is DisTorch? Standing for "distributed torch", the DisTorch nodes in this custom_node provide a way of moving the static parts of your main image generation model known as the `UNet` off your main compute card to somewhere slower, but one that is not taking up space that could be better used for longer videos or more concurrent images. By selecting one or more donor devices - main CPU DRAM or another cuda/xps device's VRAM - you can select how much of the model is loaded on that device instead of your main `compute` card. Just set how much VRAM you want to free up, and DisTorch handles the rest.

- **Two Modes**:
  - **Normal Mode**: The standard `virtual_vram_gb` slider continues to let you select one donor device (like your system's RAM) to offload to. The more virtual VRAM you add, the more of the model is pushed to the donor device. Simple and effective.
  - **Expert Mode**: For connoisseurs of performance, with two Expert Modes `byte` and `ratio` that allow you to specify exactly how the *model itself* is split across all your available devices as well as the legacy `fraction` method for *your devices* to have exact allocations. These modes are all accomplished via a single, flexible text string:
    - **Bytes (Recommended)**: The most direct way to slice up your model. Inspired by Huggingface's `device_map`, you can specify the exact number of gigabytes or megabytes for each device. The wildcard `*` assigns the remainder of the model to a device, making it easy to offload. (The CPU acts as the default wildcard if none are specified.)
      - **Example**: `cuda:0,2.5gb;cpu,*` will load the first 2.50GB of the model onto `cuda:0` and the rest onto the `cpu`.
      - **Example**: `cuda:0,500mb;cuda:1,3.0g;cpu,5gb*` will put 0.50GB on `cuda:0`, 3.00GB on `cuda:1`, and 5.00GB (or the remainder) on `cpu`.
    - **Ratio**: Love the simplicity of `llama.cpp`'s `tensor_split`? This mode is for you. Specify a ratio to distribute the model across devices.
      - **Example**: `cuda:0,25%;cpu,75%` will split the model in a 1:3 ratio, loading 25% onto `cuda:0` and 75% onto the `cpu`.
      - **Example**: `cuda:0,8%;cuda:1,8%;cpu,4%` uses an 8:8:4 ratio, putting 40% of the model on `cuda:0`, 40% on `cuda:1`, and 20% on `cpu`.
    - **Fraction**: The original DisTorch expert mode. This mode splits the model based on the fraction of each device's *total VRAM* to be used.
      - **Example**: `cuda:0,0.1;cpu,0.5` will use 10% of `cuda:0`'s VRAM and 50% of the `cpu`'s RAM to hold the model.
      - **Example**: `cuda:0,0.0207;cuda:1,0.1273;cpu,0.0808` will use 2.1% of `cuda:0`'s VRAM, 12.7% of `cuda:1`'s VRAM, and 8.1% of the `cpu`'s RAM to hold the model.

## 🎯 Key Benefits
- Free up GPU VRAM instantly without complex settings
- Run larger models by offloading layers to other system RAM
- Use all your main GPU's VRAM for actual `compute` / latent processing, or fill it up just enough to suit your needs and the remaining with quick-access model blocks.
- Seamlessly distribute .safetensors and GGUF layers across multiple GPUs if available
- Allows **you** to easily shift from ___on-device speed___ to ___open-device latent space capability___ with a simple one-number change

<p align="center">
  <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/assets/distorch_node.png" width="400">
  <br>
  <em>DisTorch Nodes with one simple number to tune its Vitual VRAM to your needs</em>
</p>

## 🚀 Compatibility
Works with all .safetensors and GGUF-quantized models.

⚙️ Expert users: Like .gguf or exl2/3 LLM loaders, use the expert_mode_alloaction for exact allocations of model shards on as many devices as your setup has!

<p align="center">
  <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/assets/distorch2_0.png" width="300">
  <br>
  <em>The new Virtual VRAM even lets you offload ALL of the model and still run compute on your CUDA device!</em>
</p>

## Installation

Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is preferred. Simply search for `ComfyUI-MultiGPU` in the list of nodes and follow installation instructions.

## Manual Installation

Clone [this repository](https://github.com/pollockjj/ComfyUI-MultiGPU) inside `ComfyUI/custom_nodes/`.

## Nodes

The extension automatically creates MultiGPU versions of loader nodes. Each MultiGPU node has the same functionality as its original counterpart but adds a `device` parameter that allows you to specify the GPU to use.

Currently supported nodes (automatically detected if available):

- Standard [ComfyUI](https://github.com/comfyanonymous/ComfyUI) model loaders:
  - CheckpointLoaderSimpleMultiGPU/CheckpointLoaderSimpleDistorch2MultiGPU
  - CLIPLoaderMultiGPU
  - ControlNetLoaderMultiGPU
  - DualCLIPLoaderMultiGPU
  - TripleCLIPLoaderMultiGPU
  - UNETLoaderMultiGPU/UNETLoaderDisTorch2MultiGPU, and 
  - VAELoaderMultiGPU
- WanVideoWrapper (requires [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)):
  - WanVideoModelLoaderMultiGPU & WanVideoModelLoaderMultiGPU_2
  - WanVideoVAELoaderMultiGPU
  - LoadWanVideoT5TextEncoderMultiGPU
  - LoadWanVideoClipTextEncoderMultiGPU
  - WanVideoTextEncodeMultiGPU
  - WanVideoBlockSwapMultiGPU
  - WanVideoSamplerMultiGPU
- GGUF loaders (requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)):
  - UnetLoaderGGUFMultiGPU/UnetLoaderGGUFDisTorch2MultiGPU
  - UnetLoaderGGUFAdvancedMultiGPU
  - CLIPLoaderGGUFMultiGPU
  - DualCLIPLoaderGGUFMultiGPU
  - TripleCLIPLoaderGGUFMultiGPU
- XLabAI FLUX ControlNet (requires [x-flux-comfy](https://github.com/XLabAI/x-flux-comfyui)):
  - LoadFluxControlNetMultiGPU
- Florence2 (requires [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2)):
  - Florence2ModelLoaderMultiGPU
  - DownloadAndLoadFlorence2ModelMultiGPU
- LTX Video Custom Checkpoint Loader (requires [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)):
  - LTXVLoaderMultiGPU
- NF4 Checkpoint Format Loader(requires [ComfyUI_bitsandbytes_NF4](https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4)):
  - CheckpointLoaderNF4MultiGPU
- HunyuanVideoWrapper (requires [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper)):
  - HyVideoModelLoaderMultiGPU
  - HyVideoVAELoaderMultiGPU
  - DownloadAndLoadHyVideoTextEncoderMultiGPU

All MultiGPU nodes available for your install can be found in the "multigpu" category in the node menu.

## Node Documentation

Detailed technical documentation is available for all **automatically-detected core MultiGPU and DisTorch2 nodes**, covering 36+ documented nodes with comprehensive parameter details, output specifications, and DisTorch2 allocation guidance where applicable.

- **To access documentation**: Click on any core MultiGPU or DisTorch2 node in ComfyUI and select "Help" (question mark inside a circle) from the resultant menu 
- **Coverage**: All standard ComfyUI loader nodes (UNet, VAE, Checkpoints, CLIP, ControlNet, Diffusers) plus popular GGUF loader variants
- **Contents**: Input parameters with data types and descriptions, output specifications, usage examples, and DisTorch2 distributed loading explanations with allocation modes and strategies
- **Note**: Documentation covers core ComfyUI-MultiGPU functionality only. Third-party custom node integrations (WanVideoWrapper, Florence2, etc.) have their own separate documentation.

## Example workflows

All workflows have been tested on a 2x 3090 + 1060ti linux setup, a 4070 win 11 setup, and a 3090/1070ti linux setup.

### DisTorch2

- [Default DisTorch2 Workflow](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/default_DisTorch2.json)
- [FLUX.1-dev Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/flux_dev_example_DisTorch2.json)
- [Hunyuan GGUF Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/hunyuan_gguf_DisTorch2.json)
- [LTX Video Text-to-Video](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/ltxv_text_to_video_MultiGPU.json)
- [Qwen Image Basic Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/qwen_image_basic_example_DisTorch2.json)
- [WanVideo 2.2 Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/wan2_2_DisTorch2.json)

### WanVideoWrapper

- [WanVideo T2V Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/wannvideowrapper/wanvideo_T2V_example_MultiGPU.json)
- [WanVideo 2.2 I2V Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/wannvideowrapper/wanvideo2_2_I2V_A14B_example_WIP_Multigpu.json)

### MultiGPU

- [FLUX.1-dev Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/multiGPU/flux_dev_example_MultiGPU.json)
- [SDXL 2-GPU](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/multiGPU/sdxl_2gpu.json)

### Florence2

- [Florence2, FLUX.1-dev, LTX Video Pipeline](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/florence2/florence2_flux1dev_ltxv_cpu_2gpu.json)

### GGUF

- [FLUX.1-dev 2-GPU GGUF](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/gguf/flux1dev_2gpu_gguf.json)
- [Hunyuan 2-GPU GGUF](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/gguf/hunyuan_2gpu_gguf.json)
- [Hunyuan CPU+GPU GGUF](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/gguf/hunyuan_cpu_1gpu_gguf.json)
- [Hunyuan GGUF DisTorch](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/gguf/hunyuan_gguf_distorch.json)
- [Hunyuan GGUF MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/gguf/hunyuan_gguf_MultiGPU.json)

### HunyuanVideoWrapper

- [HunyuanVideoWrapper Native VAE](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuanvideowrapper/hunyuanvideowrapper_native_vae.json)
- [HunyuanVideoWrapper Select Device](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/hunyuanvideowrapper/hunyuanvideowrapper_select_device.json)

### DisTorch (Legacy GGUF)

- [FLUX.1-dev GGUF DisTorch](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch/flux1dev_gguf_distorch.json)
- [Hunyuan IP2V GGUF DisTorch](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch/hunyuan_ip2v_distorch_gguf.json)

## Support

If you encounter problems, please [open an issue](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/new). Attach the workflow if possible.

## Credits

Currently maintained by [pollockjj](https://github.com/pollockjj).
Originally created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov).
With deepest thanks to [City96](https://v100s.net/).
