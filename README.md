# ComfyUI-MultiGPU v2.0.0: Universal .safetensors and GGUF Multi-GPU Distribution with DisTorch
<p align="center">
  <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/assets/distorch_average.png" width="600">
  <br>
  <em>Free almost all of your GPU for what matters: Maximum latent space processing</em>
</p>

## The Core of ComfyUI-MultiGPU v2.0.0:
[^1]: This **enhances memory management,** not parallel processing. Workflow steps still execute sequentially, but with components (in full or in part) loaded across your specified devices. *Performance gains* come from avoiding repeated model loading/unloading when VRAM is constrained. *Capability gains* come from offloading as much of the model (VAE/CLIP/UNet) off of your main **compute** device as possible—allowing you to maximize latent space for actual computation.

1.  **Universal .safetensors Support**: Native DisTorch2 distribution for all `.safetensors` models.
2.  **Up to 10% Faster GGUF Inference versus DisTorch1**: The new DisTorch2 logic provides potential speedups for GGUF models versus the DisTorch V1 method.
3.  **Bespoke WanVideoWrapper Integration**: Tightly integrated, stable support for WanVideoWrapper with eight bespoke MultiGPU nodes.

<h1 align="center">DisTorch: How It Works</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-MultiGPU/main/assets/distorch2_0.gif" width="800">
  <br>
  <em>DisTorch 2.0 in Action</em>
</p>

What is DisTorch? Standing for "distributed torch", the DisTorch nodes in this custom_node provide a way of moving the static parts of your main image generation model known as the `UNet` off your main compute card to somewhere slower, but one that is not taking up space that could be better used for longer videos or more concurrent images. By selecting one or more donor devices - main CPU DRAM or another cuda/xps device's VRAM - you can select how much of the model is loaded on that device instead of your main `compute` card. Just set how much VRAM you want to free up, and DisTorch handles the rest.

- **Virtual VRAM**: Defaults to 4GB - just adjust it based on your needs
- **Two Modes**:
  - **Donor Device**: Offloads to device of your choice, defaults to system RAM
  - **Expert Mode Allocation**: Arbitrarily assign parts of the Unet across *ALL* available devices - Fine-grained control on exactly where your models are loaded! Choose each device and what percent of that device is to be allocated for ComfyUI model loading and let ComfyUI-MultiGPU do the rest behind the scenes! 

 - Hint: Every run using the standard `virtual_vram_gb` allocation scheme creates its own v2 Expert String listed in the log.
   - **Example**:v2 Expert String cuda:0,0.2126;cpu,0.0851 = 21.26% of cuda:0 memory and 8.51% of CPU memory are dedicated to a model in this case. 
   - Play around and see how the expert string moves for your devices. You'll be custom tuning in no time!

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

## Example workflows

All workflows have been tested on a 2x 3090 + 1060ti linux setup, a 4070 win 11 setup, and a 3090/1070ti linux setup.

### DisTorch2

- [Default DisTorch2 Workflow](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/default_DisTorch2.json)
- [FLUX.1-dev Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/flux_dev_example_DisTorch2.json)
- [Hunyuan GGUF Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/hunyuan_gguf_DisTorch2.json)
- [LTX Video Text-to-Video](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/ltxv_text_to_video_MultiGPU.json)
- [Qwen Image Basic Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/qwen_image_basic_example_DisTorch2.json)
- [WanVideo 2.2 Example](https://github.com/pollockjj/ComfyUI-MultiGPU/blob/main/examples/distorch2/wan2_2_example_DisTorch2.json)

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