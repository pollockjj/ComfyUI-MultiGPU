# CPU Memory Leak Fix Plan (Updated to Current Code State)

Last updated: 2025-09-29

Executive summary
- Phase 3 (Selective Ejection) is implemented in code without the Phase 1 global sentinel.
- Current mechanism:
  - During load, DisTorch2 nodes set a per-model transient flag: `_mgpu_unload_distorch_model = (keep_loaded == False)`.
  - End-of-workflow cleanup uses ComfyUI’s standard flags (unload_models/free_memory), which route through our patched code:
    - `mm.unload_all_models` is patched to selectively unload only models where `_mgpu_unload_distorch_model == True` and retain others (rebuilds `mm.current_loaded_models` with `kept_models`).
    - `mm.soft_empty_cache` is patched to `soft_empty_cache_distorch2_patched` for multi-device VRAM clear + adaptive CPU reset, and forced `PromptExecutor.reset()` when `force=True` (Manager parity).
    - `force_full_system_cleanup()` sets both flags exactly like Manager’s “Free model and node cache”.
- Remaining defect (to fix next): In some flows, retained models are still ejected downstream. We had the selectiveness working earlier on this branch, so the next action is to rediscover and reinstate the exact working variant.

Current implementation snapshot

- Per-model transient flag (set at load time)
  - File: `distorch_2.py`
  - Where: In each DisTorch2 override (UNET/CLIP/VAE), after calling the real loader:
    - `out[0].model._mgpu_unload_distorch_model = (not keep_loaded)`
  - Purpose: Mark this model for selective ejection at unload time only if the user asked not to keep it loaded.

- Selective unload (end-of-workflow)
  - File: `model_management_mgpu.py`
  - Patch: `mm.unload_all_models` → `_mgpu_patched_unload_all_models`
  - Behavior:
    - Iterate `mm.current_loaded_models` and split into:
      - `models_to_unload`: those with `_mgpu_unload_distorch_model == True`
      - `kept_models`: everything else
    - If all models are kept (no flags set), it delegates to the original `mm.unload_all_models()`.
    - Else it unloads only `models_to_unload`, and then sets `mm.current_loaded_models = kept_models`.

- Manager parity (trigger path)
  - File: `model_management_mgpu.py`
  - `force_full_system_cleanup(reason, force=True)` sets both flags on the queue:
    - `"unload_models": True`
    - `"free_memory": True`
  - ComfyUI worker thread consumes these flags:
    - Calls `comfy.model_management.unload_all_models()` (our patched version runs)
    - Calls `PromptExecutor.reset()` when `free_memory=True`
    - Performs GC and `mm.soft_empty_cache()` (our patched version runs)

- Multi-device cache and CPU reset
  - File: `__init__.py`
  - Patch: `mm.soft_empty_cache` → `soft_empty_cache_distorch2_patched`
    - Detects if DisTorch2 is active
    - Clears VRAM on all devices via `soft_empty_cache_multigpu()`
    - Checks CPU pressure and optionally triggers executor reset (when forced)

What is not used (vs. earlier plan)
- No global executing sentinel (e.g., `DISTORCH2_UNLOAD_MODEL`). The selective logic is driven entirely by per-model `_mgpu_unload_distorch_model` flags plus the patched unload path and standard ComfyUI flags.

Observed defect (root cause unknown)
- In some flows, retained models (keep_loaded=True) are still being ejected downstream despite selective unload logic being present.
- The selective logic exists in the code and appears correct on inspection, but practical testing shows retained models are not staying loaded.

Important clarification
- The "all-kept delegation" to original `mm.unload_all_models()` when `len(kept_models) == len(mm.current_loaded_models)` is INTENTIONAL behavior.
- This delegation is necessary to trigger cleanup post-execution when no models are flagged for ejection.
- This is NOT the bug - it's required functionality for proper memory management.

Hypotheses to investigate
1) Object path mismatch in flag storage/retrieval
   - Flag may be set on one object hierarchy during load but read from a different hierarchy during unload
   - Need to verify: `out[0].model._mgpu_unload_distorch_model` vs `mp.model._mgpu_unload_distorch_model` paths match

2) Flag not persisting between load and unload
   - Something may be clearing or resetting the flag after it's set
   - Transient flag may be lost during model operations or transfers

3) Incorrect categorization logic
   - Models with keep_loaded=True being incorrectly added to `models_to_unload` instead of `kept_models`
   - Logic error in the flag evaluation or defaulting behavior

Rediscovery plan (the next step after committing this Memory Bank update)

1) Locate previously working selective retention commit(s)
   - Search this branch history for commits that logged successful retention:
     - Look for “[UNLOAD_DEBUG] Updated mm.current_loaded_models…” followed by a subsequent flow where retained models remained alive.
     - Diff the unload patch in those commits against the current `_mgpu_patched_unload_all_models` implementation.

2) Reinstate the proven selective no-op guard
   - Ensure this rule:
     - If `models_to_unload` is empty, return immediately (no-op). Do not delegate to original.
     - If `models_to_unload` is non-empty, unload only those and rebuild `mm.current_loaded_models = kept_models`.

3) Add hardening logs and assertions
   - Around unload:
     - “pre-unload snapshot”, “post-unload snapshot”, “post-reset snapshot”, “post-gc/soft_empty snapshot”.
   - If any object in `kept_models` is missing/evicted after the full free flow, log an ERROR with class name/hash.
   - Keep these until regression is confidently resolved, then demote to DEBUG if too noisy.

Verification matrix

- Minimal retention test
  - Load models: A (keep=false), B (keep=true), C (keep=true).
  - Trigger Manager-parity cleanup: unload_models=true, free_memory=true.
  - Expectation:
    - `A` is ejected. `B` and `C` remain in `mm.current_loaded_models`.
    - Memory snapshots show CPU memory decreases; VRAM caches cleared; retained models still live after the whole free flow.

- All kept test
  - Load models: D (keep=true), E (keep=true).
  - Trigger Manager-parity cleanup.
  - Expectation:
    - No models are ejected (strict no-op on unload when none are flagged).
    - Snapshots reflect cache cleaning only (allocator/torch caches), not model unloads.

Acceptance criteria

- After cleanup:
  - Only models flagged with `_mgpu_unload_distorch_model=True` are ejected.
  - Models with `_mgpu_unload_distorch_model=False` remain referenced by `mm.current_loaded_models` and alive after `PromptExecutor.reset()`, GC, and `soft_empty_cache()`.

Next steps (after this doc commit)
- Run git history to identify the prior working selective retention commit(s).
- Reinstate the working no-op behavior for the “all-kept” branch.
- Add targeted logging to confirm no retained models are ejected downstream.
- Re-run verification matrix and keep the Memory Bank synchronized.

Appendix: Relevant code touch points (as of today)
- Per-model flag: `distorch_2.py` (DisTorch2 overrides)
- Patched unload: `model_management_mgpu.py` (`mm.unload_all_models` → `_mgpu_patched_unload_all_models`)
- Patched soft empty: `__init__.py` (`mm.soft_empty_cache` → `soft_empty_cache_distorch2_patched`)
- Manager parity: `model_management_mgpu.py` (`force_full_system_cleanup` sets both queue flags)
