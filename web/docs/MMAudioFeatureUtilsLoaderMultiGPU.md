# MMAudioFeatureUtilsLoaderMultiGPU

`MMAudioFeatureUtilsLoaderMultiGPU` gathers the auxiliary MMAudio components (VAE, Synchformer, CLIP, and optional vocoder) on the device you choose so they can feed the sampler without consuming your main GPU.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `vae_model` | `STRING` | VAE weights from `ComfyUI/models/mmaudio`. |
| `synchformer_model` | `STRING` | Synchformer weights from `ComfyUI/models/mmaudio`. |
| `clip_model` | `STRING` | CLIP weights from `ComfyUI/models/mmaudio`. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `bigvgan_vocoder_model` | `VOCODER_MODEL` | Optional BigVGAN vocoder bundle. |
| `mode` | `STRING` | Feature resolution (`16k` or `44k`). |
| `precision` | `STRING` | Precision for aux weights (`fp16`, `fp32`, `bf16`). |
| `device` | `STRING` | Device receiving the feature utility stack. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `mmaudio_featureutils` | `MMAUDIO_FEATUREUTILS` | Fully prepared feature utility pack. |
