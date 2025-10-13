# MMAudioSamplerMultiGPU

`MMAudioSamplerMultiGPU` renders audio clips with MMAudio while giving you control over which accelerator runs the diffusion loop and whether frames stay offloaded.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `mmaudio_model` | `MMAUDIO_MODEL` | Core MMAudio checkpoint prepared by the loader. |
| `feature_utils` | `MMAUDIO_FEATUREUTILS` | Feature utility bundle containing VAE, Synchformer, CLIP, and optional vocoder. |
| `duration` | `FLOAT` | Target duration for the generated audio in seconds. |
| `steps` | `INT` | Number of sampler iterations to run. |
| `cfg` | `FLOAT` | Classifier-free guidance scale. |
| `seed` | `INT` | Random seed, `0` for deterministic repeatability. |
| `prompt` | `STRING` | Positive conditioning text. |
| `negative_prompt` | `STRING` | Negative conditioning text. |
| `mask_away_clip` | `BOOLEAN` | Hide supplied clip video frames during sampling. |
| `force_offload` | `BOOLEAN` | Force temporary offload of the model after sampling. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `images` | `IMAGE` | Reference frames to guide the sampler. |
| `device` | `STRING` | Device that hosts the diffusion pass (`cuda:0`, `cpu`, etc.). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `audio` | `AUDIO` | Generated audio waveform tensor. |
