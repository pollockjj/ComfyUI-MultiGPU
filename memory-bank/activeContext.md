# Active Context: Current Development Focus (Updated 2025-09-29)

## Current Work Focus

### Primary Development Status
**Project State**: Production Grade (Version 2.4.7)  
**Stability**: 300+ commits, 90 resolved issues  
**Community**: Active user base with consistent feedback  
**Performance**: Benchmarked and validated across hardware configurations  

### Recent Major Achievements (Last 6–12 Months)

#### DisTorch V2.0 Release (August 2025)
- Universal SafeTensor support (beyond GGUF)
- ~10% performance improvement over DisTorch V1
- Load-Patch-Distribute (LPD) pipeline: load on compute → patch LoRAs at full precision → distribute
- Expert allocation modes: bytes, ratios, fractions

#### City96 Architecture Integration (Dec 2024 – Ongoing)
- Code reduction: ~400 lines → ~50 lines via inheritance-based dynamic override
- Automatic node creation from existing loaders
- Maintenance simplification (fail-loudly alignment with ComfyCore API)
- Universal support for loader patterns

#### Comprehensive Hardware Validation
- 6 hardware configurations (NVLink to PCIe 3.0 x4)
- 5 model families validated (FLUX, WAN, QWEN, HunyuanVideo, Florence2)
- Clear bandwidth vs performance characterization and recommendations

## Current Development Priorities

### 1) CPU Memory Leak Resolution: Status and What’s Left
Current code state (verified in repo):
- Selective ejection (Phase 3) is implemented without the Phase 1 global sentinel.
  - During load in DisTorch2 wrappers (UNET/CLIP/VAE), we set a per-model transient flag:
    - `_mgpu_unload_distorch_model = (keep_loaded == False)`
  - End-of-workflow “free” path mirrors Manager parity by setting:
    - `unload_models=True`, `free_memory=True`
  - Patches in place:
    - `mm.unload_all_models` → selectively unloads only models with `_mgpu_unload_distorch_model == True` and rebuilds `mm.current_loaded_models` from kept models
    - `mm.soft_empty_cache` → `soft_empty_cache_distorch2_patched` (multi-device VRAM clear + adaptive CPU reset, and forceable executor reset for parity)

Outstanding defect:
- In some flows, retained (keep_loaded=True) models are still being ejected downstream.
- Two likely culprits:
  1) “All-kept delegation” in our patched unload: when no models are flagged, delegation to the original `unload_all_models()` unloads everything.
  2) Post-unload follow-on flows (e.g., `PromptExecutor.reset()`, GC, `soft_empty_cache()`, or a core `free_memory(...)` path) may cause unintended detaches for retained models.

Immediate Actions:
- Documentation sync (this update) and commit
- Rediscover the previously working selective retention variant from branch history and reinstate it
- Harden no-op path in `unload_all_models`:
  - If no models are flagged for ejection, do nothing (strict no-op), never delegate to the original
- Add temporary instrumentation:
  - Memory/log snapshots at: pre-unload → post-unload → post-reset → post-gc/soft_empty
  - ERROR if any kept model is missing after the full `/free` flow

Verification Matrix:
- Minimal retention: A(keep=false), B(true), C(true) → A ejected, B/C retained after complete free flow
- All-kept: D(true), E(true) → no ejection, only allocator/cache cleanups

Rediscovery Plan:
- Search recent commits where logs indicate successful retention after free
- Diff `_mgpu_patched_unload_all_models` vs current to recover exact guard/flow
- Confirm Manager parity (`/free` flags) still routes through patched unload and retains kept models across reset/GC

### 2) Ecosystem Expansion (High Priority)
Goal: Support emerging model formats and custom nodes

Active Integrations:
- ComfyUI-GGUF: DisTorch-enabled GGUF nodes (complete)
- WanVideoWrapper: MultiGPU video nodes (complete)
- Florence2: Vision model support (complete)
- HunyuanVideoWrapper: Native VAE + device selection (in progress)

Next Targets:
- LTX Video
- Mochi
- Issue-driven community requests

### 3) User Experience Optimization (Medium Priority)
Goal: Reduce complexity while preserving expert control

Recent Improvements:
- Automatic Mode: Intelligent offloading based on VRAM availability
- Error messages: Clearer guidance for allocation failures
- Documentation: 20+ example JSON workflows

Ongoing:
- Configuration validation and performance prediction
- “First-run” guides for low-VRAM and multi-GPU users

### 4) Advanced Features (Low Priority)
Research Areas:
- Model parallelism and pipeline parallelism
- Memory compression, fragmentation handling
- Quality metrics and deterministic parity checks

## Active Technical Decisions

### Memory Management Philosophy
- Conservative by default with explicit user control
- Preserve quality: Patch LoRAs before distributing
- Transparency: Verbose and structured memory logging
- Fail-loudly alignment with ComfyCore

### Integration Strategy
- Inheritance-based node override (City96 pattern)
- Minimal patch surface area with explicit patch points:
  - `mm.get_torch_device`/`mm.text_encoder_device` override for device selection
  - `mm.soft_empty_cache` override for multi-device cache clear + CPU reset
  - `mm.unload_all_models` selective unload path

### Hardware Support Priority
- Tier 1: CUDA
- Tier 2: CPU, MPS
- Tier 3: XPU, NPU, MLU, DirectML (experimental footprint grows with community validation)

## User Behavior Patterns (Observed)
- Low-VRAM image gen, multi-GPU video gen, professional pipelines, enthusiast experiments
- Support requests: device detection, OOM, performance expectations, missing nodes, quality concerns, integration requests
- Allocation preferences: bytes (most common), fraction, ratio

## Next Steps & Immediate Actions
Short-term (2–4 weeks):
- Commit Memory Bank updates (this change)
- Rediscover and reinstate the selective retention behavior that worked
- Harden no-op branch in unload patch and add retention instrumentation
- Run verification matrix and update docs with results
- Triage top GitHub issues

Medium-term (2–3 months):
- LTX Video integration
- Performance dashboard and quality measurement runs
- Tutorials and doc refresh based on latest capabilities

Long-term (6–12 months):
- Model/pipeline parallelism experiments
- Streaming inference for video
- Multi-node/cloud integration and orchestration

## Current Environment State
- IDE: VSCode
- Version Control: Git with conventional commits
- Testing: Manual validation on available hardware + community contributions
- Primary Dev HW: RTX 3090 + mixed secondaries
- Known Limitation: Limited access to newest GPUs (e.g., RTX 5090)

This Active Context reflects the current codebase reality: Phase 3 selective ejection is in place (per-model flags + selective unload patch), but a retention defect remains when no models are flagged and/or after the free path completes. The immediate roadmap is to commit these updates, then locate and reinstate the previously working selective retention behavior and add guards to ensure robust “keep_loaded=True” semantics across the full Manager parity flow.
