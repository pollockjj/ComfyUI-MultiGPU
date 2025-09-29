# Technical Context & Dependencies (Updated 2025-09-29)

## Core Technology Stack

### Python Environment
Requirements:
- Python 3.10+ recommended
- PyTorch 2.x (CUDA/HIP/XPU backends as available)
- ComfyUI as host framework

### Framework Dependencies

Required (ComfyUI Core)
```python
import torch
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths
```

Optional (External Custom Nodes)
```python
# ComfyUI-GGUF Integration
try:
    from ComfyUI_GGUF import nodes as gguf_nodes
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

# WanVideoWrapper Integration  
try:
    import ComfyUI_WanVideoWrapper.nodes as wanvideo_nodes
    WANVIDEO_AVAILABLE = True
except ImportError:
    WANVIDEO_AVAILABLE = False
```

## Device Support Matrix

Primary Support (tested)
- CUDA (NVIDIA)
- CPU
- MPS (Apple Metal)

Extended/Community
- XPU (Intel)
- NPU (Ascend)
- MLU (Cambricon)
- DirectML (Windows)
- CoreX/IXUCA

## Integration Architecture (Current Patch Points)

This project extends ComfyUI through carefully scoped patches and runtime overrides. The current core integration points are:

1) get_torch_device/text_encoder_device override (device selection)
- File: `__init__.py`
- Patch:
  - `mm.get_torch_device = get_torch_device_patched`
  - `mm.text_encoder_device = text_encoder_device_patched`
- Purpose: Respect user-selected devices handoff by MultiGPU wrappers and maintain ComfyUI alignment.

2) soft_empty_cache (multi-device + CPU reset)
- File: `__init__.py`
- Patch:
  - `mm.soft_empty_cache = soft_empty_cache_distorch2_patched`
- Behavior:
  - Detects DisTorch2 activity, clears allocator caches across ALL devices via `soft_empty_cache_multigpu()` (from `device_utils.py`)
  - Adaptive CPU memory reset (threshold-based), and optional forced `PromptExecutor.reset()` when `force=True` (Manager parity)

3) unload_all_models (selective ejection)
- File: `model_management_mgpu.py`
- Patch:
  - `mm.unload_all_models = _mgpu_patched_unload_all_models`
- Behavior:
  - Splits `mm.current_loaded_models` into:
    - `models_to_unload` where per-model `_mgpu_unload_distorch_model == True`
    - `kept_models` for all others
  - If flagged models exist: unload them only, then set `mm.current_loaded_models = kept_models`
  - Current caveat: When none are flagged, the code delegates to the original unload (target is strict no-op; see System Patterns and Fix Plan)

4) DisTorch2 load-time model flagging (per-model transient)
- File: `distorch_2.py`
- Where:
  - In DisTorch2 wrappers (UNET/CLIP/VAE) within `override(...)` after original call:
    - `out[0].model._mgpu_unload_distorch_model = (keep_loaded == False)`
- Rationale:
  - Surgical per-model control enables selective ejection in patched unload without any global sentinel

5) Manager parity helper
- File: `model_management_mgpu.py`
- Function:
  - `force_full_system_cleanup(reason="manual", force=True)`
- Behavior:
  - Sets both `unload_models=True` and `free_memory=True` on PromptQueue, matching Manager’s “Free model and node cache” button behavior

## Selective Ejection Flow (Technical Overview)

- Load time (DisTorch2 wrappers):
  - Mark models for ejection if keep_loaded=False
- Free flow (Manager or programmatic parity):
  - /free → prompt_worker picks flags → calls `mm.unload_all_models()` (selective) → `PromptExecutor.reset()` → GC → `mm.soft_empty_cache()` (multi-device)
- Intended properties:
  - Models flagged for ejection are destroyed
  - Retained models remain live after full flow (including reset/GC/soft_empty)

Current caveat (to fix next):
- When no models are flagged, the patched unload delegates to the original unload, which unloads everything. The target is strict no-op in this branch.

## Development Environment

Supported OS
- Linux (primary)
- Windows 10/11
- macOS (Apple Silicon via MPS)

Tools
- IDE: VSCode
- VCS: Git (conventional commits encouraged)
- Testing: Manual validation across available hardware + community testing

## Performance Characteristics

Bandwidth hierarchy
1. NVLink (~50.8 GB/s) – near-native performance
2. PCIe 4.0 x16 (~27.2 GB/s) – excellent offloading
3. PCIe 3.0 x8 (~6.8 GB/s)
4. PCIe 3.0 x4 (~2.1 GB/s)

Load-Patch-Distribute (LPD)
- Always load on compute device first
- Apply LoRAs at full precision
- Distribute blocks to assigned devices for final placement
- Ensures quality preservation and deterministic behavior

## Configuration Management

Expert allocation strings
- Bytes mode (recommended):
  - `"cuda:0,2.5gb;cuda:1,3.0g;cpu,*"`
- Ratio mode:
  - `"cuda:0,25%;cpu,75%"`
- Fraction mode (legacy):
  - `0.8`, `0.5`, `0.95`

## Debugging & Monitoring

Logging
- `logger.mgpu_mm_log(...)` for structured memory/system logs
- `multigpu_memory_log(identifier, tag)` for timestamped CPU/VRAM snapshots

Inspection
- `device_utils.comfyui_memory_load(tag)` for one-line current memory snapshot
- VRAM cache clearing logs around `soft_empty_cache_multigpu()`

## Architectural Rationale (Updated)

Per-model flag over global sentinel
- Granular control, no persistent global state
- Isolated to each loaded model, matches ComfyUI lifecycle

Patched unload behavior (selective)
- Maintain `kept_models` across the full free path
- Only eject DisTorch2 models when explicitly requested via keep_loaded=False

Patched soft empty (multi-device)
- Ensure cache clearing is not limited to the single `mm.get_torch_device()` device
- CPU memory behavior integrated with PromptExecutor.reset() semantics

## Known Technical Work (Next)

- Reinstate strict no-op in `_mgpu_patched_unload_all_models` when `models_to_unload` is empty (no delegation to original unload)
- Add instrumentation and assertions to guarantee no unintended ejection of retained models after `/free` flow
- Re-run verification matrix and capture logs in Memory Bank
