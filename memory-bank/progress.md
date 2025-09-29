# Project Progress & Status (Updated 2025-09-29)

## What Works (Production Ready)

### Core MultiGPU Infrastructure ✅
- Dynamic Class Override System (City96): inheritance-based node wrapping, auto-adapts to ComfyCore
- Device Detection: CPU, CUDA, MPS, XPU, NPU, MLU, DirectML, CoreX
- VRAM Management: Multi-device cache clearing via `soft_empty_cache_multigpu`
- Node Registration: Automatic node creation based on available dependencies

### DisTorch2 Distributed Loading ✅
- Universal SafeTensor support (beyond GGUF)
- Load-Patch-Distribute pipeline (quality-preserving LoRA patching on compute device)
- Expert allocation modes (bytes, ratios, fractions)
- ~10% performance improvement over DisTorch V1

### Selective Unloading (Implemented) ✅
- Per-model transient flag is set by DisTorch2 loader wrappers:
  - `_mgpu_unload_distorch_model = (keep_loaded == False)`
- Patched unload path:
  - `mm.unload_all_models` → selectively unloads models with `_mgpu_unload_distorch_model=True` and rebuilds `mm.current_loaded_models` with retained models
- Patched soft empty:
  - `mm.soft_empty_cache` → `soft_empty_cache_distorch2_patched`: multi-device allocator cache clearing + adaptive CPU reset; can force executor reset for Manager parity
- Manager parity helper:
  - `force_full_system_cleanup` sets `unload_models` and `free_memory` flags to mirror the “Free model and node cache” button

### Hardware Configuration Support ✅
- NVLink: near-native performance
- PCIe 4.0 CPU offloading: excellent performance
- Legacy hardware: PCIe 3.0 coverage with documented trade-offs
- Mixed architectures: supported

### External Integrations ✅
- ComfyUI-GGUF: DisTorch-enabled quantized model nodes
- WanVideoWrapper: MultiGPU video nodes
- Florence2: Vision model support
- HunyuanVideoWrapper: Native VAE + device selection (active)

### Documentation & Examples ✅
- Comprehensive README
- 20+ example workflows
- Performance benchmarks and configuration recommendations

## What’s Left to Build (Development Roadmap)

### Short-term Enhancements (Next 2–4 weeks)

#### Selective Retention Hardening (Top Priority) 🔄
- Current state:
  - Phase 3 selective ejection implemented without global sentinel
  - In some flows, retained models (keep_loaded=True) are still ejected downstream
- Likely culprits:
  1) “All-kept delegation” in patched unload: when no models are flagged, current code delegates to original unload which unloads everything
  2) Post-unload follow-on flows (PromptExecutor.reset/GC/soft_empty/free_memory path) may detach retained models
- Action plan:
  - Rediscover prior commit(s) where selectiveness worked end-to-end
  - Reinstate strict no-op when `models_to_unload` is empty (do not delegate to original)
  - Add instrumentation: pre/post unload → post reset → post GC/soft_empty snapshots; ERROR if any kept model disappears
  - Re-run verification matrix (A=false, B/C=true; D/E all kept)

#### User Experience Improvements 🔄
- Configuration validation and performance prediction
- Refined error messaging for allocation/placement issues
- Documentation refresh for current state (this update)

#### Integration Expansion 🔄
- LTX Video support
- Mochi integration
- Issue-driven community requests

### Medium-term Goals (2–3 months)

#### Advanced Memory Management 📋
- Memory compression / fragmentation handling research
- Enhanced retention/eviction policies under pressure
- Robust regression tests for retention across `/free` flow

#### Professional Features 📋
- Batch processing tooling
- API server modes for automation
- Quality metrics and reproducibility checks
- Performance dashboard

#### Community Tools 📋
- Allocation string generator w/ validation
- Hardware profiler (bandwidth/VRAM/latency)
- Compatibility matrix (community-maintained)
- Tutorials and video guides

### Long-term Research (6–12 months)

#### Next-Generation Features 🔬
- Model parallelism and pipeline parallelism
- Streaming inference for video
- Multi-node/cloud distributed inference
- Deterministic output equivalence verification

## Current Status Assessment

### Stability: Production Grade (8/10)
- CPU memory leak: Phase 3 implemented, retention bug remains in some flows
- Crash rate: Low based on community feedback
- API compatibility: Stable with ComfyCore
- Hardware coverage: Broad and documented

### Performance: Optimized (8/10)
- NVLink: 5–7% slowdown vs native in typical cases
- PCIe 4.0 CPU offloading: ~40–50% slowdown with excellent price/perf
- Predictable tradeoffs based on bandwidth hierarchy

### Feature Completeness: Comprehensive (8.5/10)
- Core functionality: Implemented
- Model support: Major families (FLUX, WAN, QWEN, etc.)
- Hardware support: Universal
- UX: Good docs/examples; ongoing improvement

### Community Adoption: Growing (7/10)
- Active stars/issues/discussions
- Integration requests from other node ecosystems
- Positive feedback with actionable feature requests

## Known Issues & Limitations

### Selective Retention Bug 🐛
- Symptom: Retained models (keep_loaded=True) sometimes ejected during `/free`
- Cause suspects:
  - All-kept delegation to original unload
  - Post-unload flows (reset/GC/soft_empty/free_memory)
- Status: High priority; rediscovery and hardening planned

### ComfyUI API Dependencies
- Core changes can impact patch points
- Fail-loudly approach surfaces issues quickly
- Ongoing monitoring required

### Hardware Edge Cases
- Exotic configurations may need targeted validation
- System RAM bandwidth can impact offloading performance

### Documentation Gaps
- Hardware selection and configuration recipes (ongoing)
- Edge-case troubleshooting

## Evolution of Project Decisions (Highlights)

- Dynamic class override over manual node duplication
- Load-Patch-Distribute over direct distribution
- Per-model unload flag over global sentinel
- Fail-loudly over defensive abstraction

## Success Metrics & Validation

### Technical
- Zero regressions in selective retention tests
- Predictable performance across bandwidth tiers
- Quality parity with single-GPU baselines

### User
- Previously impossible workflows now run reliably
- Clear guidance for low-VRAM and multi-GPU users
- Reduced support load for common issues

### Ecosystem
- Broader adoption in custom node projects
- Recognition in optimization discussions
- Community contributions to validation

## Next Steps (Actionable)
- Commit Memory Bank sync (this change)
- Git archeology to recover working selective retention diff
- Implement strict no-op for all-kept branch in unload
- Add temporary instrumentation; run verification matrix
- Update docs with results and remove extra logs after stabilization
