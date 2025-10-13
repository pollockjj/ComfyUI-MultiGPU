# PulidInsightFaceLoaderMultiGPU

`PulidInsightFaceLoaderMultiGPU` boots the InsightFace detector needed by PuLID and pins it to the device you specify, ensuring face embeddings come from the best accelerator for your setup.

## Inputs

### Required

| Parameter | Data Type | Description |
| --- | --- | --- |
| `provider` | `STRING` | Execution backend (`CPU`, `CUDA`, `ROCM`, `CoreML`). |

### Optional

| Parameter | Data Type | Description |
| --- | --- | --- |
| `device` | `STRING` | Device assigned to the InsightFace runtime. |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `faceanalysis` | `FACEANALYSIS` | Ready InsightFace analysis module. |
