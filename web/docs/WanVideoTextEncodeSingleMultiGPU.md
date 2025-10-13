# WanVideoTextEncodeSingleMultiGPU

`WanVideoTextEncodeSingleMultiGPU` encodes a single prompt string (no negative branch) using a Wan T5 encoder while honouring the device you supply. Use it for LoRA control channels or scenarios where only one conditioning embedding is required.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `prompt` | `STRING` | Prompt text to encode. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `t5` | `WANTEXTENCODER` | Encoder pair from `LoadWanVideoT5TextEncoderMultiGPU`. |
| `load_device` | `MULTIGPUDEVICE` | Device that will perform encoding. |
| `force_offload` | `BOOLEAN` | Offload linked models after encoding completes. |
| `model_to_offload` | `WANVIDEOMODEL` | Wan diffusion model to move while encoding to free VRAM. |
| `use_disk_cache` | `BOOLEAN` | Store/reuse embeddings on disk for repeat runs. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `text_embeds` | `WANVIDEOTEXTEMBEDS` | Encoded embeddings ready for Wan samplers. |
