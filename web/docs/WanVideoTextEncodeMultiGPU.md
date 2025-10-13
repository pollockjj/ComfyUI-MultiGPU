# WanVideoTextEncodeMultiGPU

`WanVideoTextEncodeMultiGPU` encodes paired positive/negative prompts using a WanVideo T5 encoder while respecting the MultiGPU device you choose. The node can temporarily offload Wan models to free VRAM before encoding and supports optional disk caching for embeddings.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `positive_prompt` | `STRING` | Prompt text for the conditioned branch. |
| `negative_prompt` | `STRING` | Prompt text for the unconditioned branch. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `t5` | `WANTEXTENCODER` | Encoder pair from `LoadWanVideoT5TextEncoderMultiGPU`; defaults to the global Wan encoder if omitted. |
| `load_device` | `MULTIGPUDEVICE` | Device to run encoding on; also controls temporary model moves. |
| `force_offload` | `BOOLEAN` | When true, offloads the model after encoding completes. |
| `model_to_offload` | `WANVIDEOMODEL` | Wan diffusion model to move to the offload device prior to encoding. |
| `use_disk_cache` | `BOOLEAN` | Enable Wan disk cache for repeated prompt reuse. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `text_embeds` | `WANVIDEOTEXTEMBEDS` | Dictionary of positive and negative embeddings ready for `WanVideoSamplerMultiGPU`. |
