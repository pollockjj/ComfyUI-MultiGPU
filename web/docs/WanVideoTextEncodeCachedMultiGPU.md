# WanVideoTextEncodeCachedMultiGPU

`WanVideoTextEncodeCachedMultiGPU` is a convenience wrapper that loads a Wan T5 encoder on demand, produces prompt embeddings, and fully unloads the encoder when finished. It favours disk caching so repeated prompts can reuse saved embeddings without re-running the model.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_name` | `STRING` | T5 encoder to load from `ComfyUI/models/text_encoders`. |
| `precision` | `STRING` | Precision for the temporary encoder (`fp32` or `bf16`). |
| `positive_prompt` | `STRING` | Prompt text for the conditioned branch. |
| `negative_prompt` | `STRING` | Prompt text for the unconditioned branch. |
| `quantization` | `STRING` | FP8 switch (`disabled` or `fp8_e4m3fn`). |
| `use_disk_cache` | `BOOLEAN` | Enables Wan disk caching for embeddings. |
| `load_device` | `STRING` | MultiGPU device that will host the one-shot encoder. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `extender_args` | `WANVIDEOPROMPTEXTENDER_ARGS` | Configuration for Wan prompt extender helpers. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `text_embeds` | `WANVIDEOTEXTEMBEDS` | Positive/negative embedding bundle for Wan samplers. |
| `negative_text_embeds` | `WANVIDEOTEXTEMBEDS` | Negative-only embeddings (for workflows that split branches). |
| `positive_prompt` | `STRING` | The positive prompt as finalised by the extender (handy for preview nodes). |
