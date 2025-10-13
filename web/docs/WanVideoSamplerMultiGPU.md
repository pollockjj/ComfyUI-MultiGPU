# WanVideoSamplerMultiGPU

`WanVideoSamplerMultiGPU` runs WanVideo diffusion sampling while respecting your chosen compute and offload devices. The node patches WanVideo's internal device tracking so samplers, transformer blocks, and optional block swap features all target the MultiGPU placements you configure.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model` | `WANVIDEOMODEL` | Wan diffusion model output by `WanVideoModelLoaderMultiGPU`. |
| `compute_device` | `MULTIGPUDEVICE` | Device identifier to run the sampler on. |
| `image_embeds` | `WANVIDIMAGE_EMBEDS` | Latent/video conditioning produced by Wan preprocessing nodes. |
| `steps` | `INT` | Number of denoising steps to execute. |
| `cfg` | `FLOAT` | Classifier-free guidance strength. |
| `shift` | `FLOAT` | Scheduler-specific shift parameter. |
| `seed` | `INT` | Random seed for reproducibility (0 uses the provided value). |
| `force_offload` | `BOOLEAN` | When true, move the model back to the offload device after sampling. |
| `scheduler` | `STRING` | Sampler scheduler to use (`unipc`, `dpm++`, `euler`, etc.). |
| `riflex_freq_index` | `INT` | Enables RIFLEX continuation frames when > 0. |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `text_embeds` | `WANVIDEOTEXTEMBEDS` | Conditioning from Wan text encoders. |
| `samples` | `LATENT` | Initial latents for video-to-video workflows. |
| `denoise_strength` | `FLOAT` | Fraction of steps to apply when reusing latents. |
| `feta_args` | `FETAARGS` | Wan FETA extension controls. |
| `context_options` | `WANVIDCONTEXT` | Context window adjustments. |
| `cache_args` | `CACHEARGS` | Cache behaviour for incremental runs. |
| `flowedit_args` | `FLOWEDITARGS` | FlowEdit animation refinements. |
| `batched_cfg` | `BOOLEAN` | Batch cond/uncond passes to trade VRAM for speed. |
| `slg_args` | `SLGARGS` | Sparse latent guidance options. |
| `rope_function` | `STRING` | Rotary embedding mode (`default`, `comfy`, `comfy_chunked`). |
| `loop_args` | `LOOPARGS` | Looping schedule configuration. |
| `experimental_args` | `EXPERIMENTALARGS` | Wan experimental toggles. |
| `sigmas` | `SIGMAS` | Custom sigma schedule. |
| `unianimate_poses` | `UNIANIMATE_POSE` | Pose conditioning inputs. |
| `fantasytalking_embeds` | `FANTASYTALKING_EMBEDS` | Speech animation embeds. |
| `uni3c_embeds` | `UNI3C_EMBEDS` | Multi-character conditioning embeds. |
| `multitalk_embeds` | `MULTITALK_EMBEDS` | MultiTalk conditioning embeds. |
| `freeinit_args` | `FREEINITARGS` | FreeInit configuration. |
| `start_step` | `INT` | Start step for partial denoising. |
| `end_step` | `INT` | End step for partial denoising (-1 uses full schedule). |
| `add_noise_to_samples` | `BOOLEAN` | Adds fresh noise to latents before diffusion. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `samples` | `LATENT` | Final latent video tensor after sampling. |
| `denoised_samples` | `LATENT` | Optional mid-run denoised latents for reuse or decoding. |
