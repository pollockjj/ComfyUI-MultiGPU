# ComfyUI Core Lineage & Integration Analysis (Updated 2025-09-29)

## Overview

ComfyUI‑MultiGPU extends (does not replace) ComfyUI core. Principles:
- Extend, not replace: patch specific core functions and inherit existing nodes
- Fail loudly: small, explicit patch points so core API changes surface quickly
- User agency: device placement is explicit and honored
- Multi‑device native: treat all devices as first‑class

Current code reality:
- Phase 3 “Selective Ejection” is implemented via a per‑model flag (no global sentinel).
- Outstanding caveat: when no models are flagged, the current unload path delegates to the original core unload (unloads everything). Target is strict no‑op in this branch.

## ComfyUI Core Foundation (Reference)

Key concepts implemented by ComfyUI core (see memory-bank/comfy_core.py snapshot):
- Global list: `current_loaded_models`
- Model wrapper: `LoadedModel` with methods like `model_load`, `model_unload`, `model_memory_required`
- Memory utilities: `soft_empty_cache()`, `get_free_memory()`, etc.
- Prompt execution:
  - `/free` endpoint sets queue flags: `unload_models`, `free_memory` (server.py)
  - `main.py` prompt worker consumes flags:
    - If `unload_models` (or `free_memory`): `comfy.model_management.unload_all_models()`
    - If `free_memory`: `PromptExecutor.reset()`
    - Then GC + `comfy.model_management.soft_empty_cache()`

This is the canonical “Manager button” path for model + execution cache cleanup.

## How MultiGPU Extends ComfyUI Core

MultiGPU adds small patches and inherits nodes to enable multi‑device behavior while preserving ComfyUI’s flow.

### 1) Device selection alignment
- File: `__init__.py`
- Patches:
  - `mm.get_torch_device = get_torch_device_patched`
  - `mm.text_encoder_device = text_encoder_device_patched`
- Purpose: Respect user‑selected devices supplied by MultiGPU wrappers while staying coherent with ComfyUI’s device model.

### 2) Multi‑device VRAM cache + CPU reset
- File: `__init__.py`
- Patch:
  - `mm.soft_empty_cache = soft_empty_cache_distorch2_patched`
- Behavior:
  - Detects if any DisTorch2 model is active and clears allocator caches on ALL devices via `soft_empty_cache_multigpu()` (from `device_utils.py`)
  - Integrates adaptive CPU memory reset; can force `PromptExecutor.reset()` on `force=True` for Manager parity

### 3) Selective ejection (patched unload)
- File: `model_management_mgpu.py`
- Patch:
  - `mm.unload_all_models = _mgpu_patched_unload_all_models`
- Behavior:
  - Iterate `mm.current_loaded_models` and split into:
    - `models_to_unload`: models with per‑model flag `_mgpu_unload_distorch_model == True`
    - `kept_models`: all others
  - If any flagged: unload only the flagged models and set `mm.current_loaded_models = kept_models`
  - Current caveat: If none are flagged (all kept), code delegates to original core unload, which unloads everything (target: strict no‑op for this branch)

### 4) Per‑model flag is set at load time (no global sentinel)
- File: `distorch_2.py`
- Where:
  - In DisTorch2 wrappers (UNET/CLIP/VAE) inside `override(...)`, after calling the original loader:
    - `out[0].model._mgpu_unload_distorch_model = (keep_loaded == False)`
- Rationale:
  - Surgical precision at model granularity and no persistent global state

### 5) Manager parity helper for tests/flows
- File: `model_management_mgpu.py`
- Function:
  - `force_full_system_cleanup(reason="manual", force=True)`
- Behavior:
  - Sets both `unload_models=True` and `free_memory=True` on the PromptQueue, just like the Manager “Free model and node cache” button

## End‑to‑End Free Flow (Now)

“Manager button” or parity helper triggers the same core actions:

1) POST /free with `{"unload_models": true, "free_memory": true}`
2) `main.py` prompt worker consumes flags:
   - Calls `comfy.model_management.unload_all_models()`
     - MultiGPU patched unload runs:
       - If any models flagged via `_mgpu_unload_distorch_model=True`: unload only those and retain others
       - If none are flagged: current code delegates to original unload (unloads everything) — under review
   - Calls `PromptExecutor.reset()`
   - GC + `comfy.model_management.soft_empty_cache()`
     - MultiGPU patched soft empty runs:
       - Multi‑device allocator cache clear (CUDA/MPS/XPU/NPU/MLU/DirectML/CoreX as available)
       - Optional CPU reset behavior when forced

Intended invariant (target):
- Only flagged DisTorch2 models are ejected; unflagged (keep_loaded=True) models remain live after the full flow.

## Behavior Notes & Next Step

- Implemented:
  - Per‑model selective ejection (Phase 3) without global sentinel
  - Multi‑device allocator clearing and Manager parity semantics
- Caveat:
  - If no models are flagged, current patched unload delegates to original unload (unloads everything)
  - This can defeat selectiveness when all models are intended to be retained
- Next step (hardening):
  - Reinstate “strict no‑op” in the all‑kept branch of `_mgpu_patched_unload_all_models` (never delegate to original unload if nothing is flagged)
  - Add instrumentation around pre/post unload, post reset, post soft‑empty to ensure retained models remain alive

## Sequence Summary

A) Vanilla ComfyUI Manager “Free
