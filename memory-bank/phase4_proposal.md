# Phase 4: Post-Execution Hook Architecture for Selective Model Cleanup

**Created:** 2025-09-29  
**Status:** Proposal under evaluation  
**Context:** Alternative to Phase 3's flag-based selective unload approach

## Executive Summary

Phase 4 proposes patching ComfyUI's `PromptExecutor.execute_async()` to add a post-execution cleanup hook using a WeakSet registry. This represents a fundamental shift from Phase 3's approach of patching `unload_all_models()` within ComfyUI's existing cleanup flow.

**Key Difference:** Phase 4 controls WHEN cleanup happens (via execute_async finally block) rather than HOW it happens (via selective unload patch).

## Background: Why Phase 3's Timing Is Actually Correct

### The Critical Misunderstanding About `force_full_system_cleanup()`

Initial analysis incorrectly assumed calling `force_full_system_cleanup()` during load meant cleanup happened DURING execution. This is wrong.

**Actual Flow (Verified from ComfyUI Core):**

1. During workflow execution, DisTorch nodes call:
   ```python
   if unload_distorch_model:
       force_full_system_cleanup(reason="policy_every_load", force=True)
   ```

2. This sets flags on the queue:
   ```python
   pq.set_flag("unload_models", True)
   pq.set_flag("free_memory", True)
   ```

3. **Flags are DEFERRED** - they sit in queue until execution completes

4. Post-execution (from `main.py`):
   ```python
   # AFTER e.execute() returns and prompt completes:
   flags = q.get_flags()
   
   if flags.get("unload_models", free_memory):
       comfy.model_management.unload_all_models()  # Runs AFTER execution
   
   if free_memory:
       e.reset()  # Clears execution caches
   
   if need_gc:
       gc.collect()
       comfy.model_management.soft_empty_cache()
   ```

**Evidence from user's log:**
```
Prompt executed in 38.53 seconds
[Phase 2 Debug] Patched unload_all_models called
```

The unload happens AFTER "Prompt executed" - proving the timing is already post-execution.

### The Graveyard of In-Execution Attempts

User's commit history reveals multiple failed attempts (Sept 9-11, 2025):
- "Improve memory handling for safetensor models"
- "Additional garbage/cache collection"
- Then: "Hot Fix: Revert aggressive memory management" (caused OOMs)
- "roll back aggressive memory management"

**Why they failed:** Attempting cleanup DURING execution when models are:
- Wrapped in weakrefs by ComfyUI
- Locked/protected during execution
- Inaccessible for cleanup operations

**User quote:** "This entire branch is a graveyard of ineffectual memory management because I am attempting to do all of it DURING execution most operations simply did nothing or were prevented because everything is instantly weakref'd the moment they spring into existence until execution is complete."

## ComfyUI-Manager Approach (The Benchmark)

### JavaScript Button Implementation
```javascript
// From common.js
mode = '{"unload_models": true, "free_memory": true}';
api.fetchApi(`/free`, {
    method: 'POST',
    body: mode
});
```

### Backend Processing
The `/free` endpoint sets both flags, which are consumed post-execution exactly like DisTorch's current approach.

**Key Insight:** Manager's "Free model and node cache" button uses THE EXACT SAME MECHANISM as Phase 3:
- Sets `unload_models=True` and `free_memory=True` flags
- Flags are processed post-execution
- Triggers the same cleanup flow

## WanVideoWrapper Approach (Direct Calls)

### Pattern Found
```python
# From nodes_sampler.py line ~600
mm.unload_all_models()
mm.soft_empty_cache()
gc.collect()
```

**Critical Difference:** WanVideoWrapper calls these DIRECTLY within their node execution function (synchronous). This works because:
- They control the exact timing within their own execution
- They call at strategic points (before sampling, after offload)
- They're not trying to be selective - they unload EVERYTHING

**Why This Doesn't Apply to DisTorch:**
- DisTorch needs SELECTIVE unloading (keep some, unload others)
- DisTorch models are managed by ComfyUI's global `current_loaded_models` list
- Direct manipulation during execution would conflict with ComfyUI's tracking

## Phase 4 Proposal: Detailed Architecture

### Core Mechanism

Patch `execution.PromptExecutor.execute_async()` to add guaranteed post-execution cleanup:

```python
# New module: distorch_lifecycle.py
import weakref
import execution
from comfy.model_patcher import ModelPatcher

_models_to_unload_post_execution = weakref.WeakSet()

def register_for_cleanup(model_patcher):
    """Called by DisTorch nodes during load with keep_loaded=False"""
    if isinstance(model_patcher, ModelPatcher):
        _models_to_unload_post_execution.add(model_patcher)

_original_execute_async = execution.PromptExecutor.execute_async

async def _patched_execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
    _models_to_unload_post_execution.clear()
    
    try:
        # Original execution
        await _original_execute_async(self, prompt, prompt_id, extra_data, execute_outputs)
    finally:
        # GUARANTEED post-execution cleanup
        if _models_to_unload_post_execution:
            for model_patcher in list(_models_to_unload_post_execution):
                _selective_unload_instance(model_patcher)
            
            # Comprehensive cleanup
            mm.soft_empty_cache()
            gc.collect()
        
        _models_to_unload_post_execution.clear()
```

### Integration Points

**DisTorch Nodes (distorch_2.py):**
```python
# In override() method:
if not keep_loaded:
    register_for_cleanup(out[0])  # Register the ModelPatcher
```

## Critical Evaluation: Phase 4 vs Phase 3

### Timing Comparison

| Aspect | Phase 3 (Current) | Phase 4 (Proposed) |
|--------|-------------------|-------------------|
| **Trigger Point** | Flag set during execution → processed post-execution | `finally` block in execute_async |
| **Actual Cleanup Time** | POST-execution (after prompt completes) | POST-execution (after prompt completes) |
| **Guarantee Level** | Depends on flag processing | Guaranteed by finally block |

**CRITICAL FINDING:** Both run at the SAME time (post-execution). Phase 3's timing is already correct.

### Architectural Comparison

| Feature | Phase 3 | Phase 4 |
|---------|---------|---------|
| **Patch Point** | `unload_all_models()` | `execute_async()` |
| **Invasiveness** | Medium (hooks into cleanup) | High (hooks into execution core) |
| **Comfy Integration** | Uses native flag system | Bypasses flag system |
| **Failure Handling** | Relies on Comfy's error flow | Guaranteed via finally |
| **State Tracking** | Per-model flags | WeakSet registry |
| **Detection Logic** | Flag checking in unload | Direct instance tracking |

### Advantages of Phase 4

1. **Zero Ambiguity:** WeakSet registry eliminates flag detection issues
   - No object path mismatches
   - No flag persistence concerns
   - Direct instance tracking

2. **Guaranteed Execution:** `finally` block runs even if execution fails

3. **Cleaner Separation:** Doesn't fight ComfyUI's unload logic, adds parallel cleanup

4. **Explicit Control:** Exactly when and what gets unloaded is deterministic

### Disadvantages of Phase 4

1. **More Invasive:** Patches core execution flow (higher risk)

2. **Bypasses ComfyUI Patterns:** Doesn't use native flag system

3. **Direct State Manipulation:** Removes from `mm.current_loaded_models` directly
   - Could cause state inconsistencies with ComfyUI's internal tracking
   - Risk of memory leaks if ComfyUI holds other references

4. **Duplicate Cleanup:** Runs IN ADDITION to ComfyUI's normal cleanup flow
   - Flag-triggered cleanup still happens
   - Could cause conflicts or double-processing

## The Actual Problem (Not Solved by Phase 4)

**Phase 3's bug is NOT about timing** - both approaches run post-execution.

**The real bug:** Selective retention logic exists and appears correct, but retained models (keep_loaded=True) are still ejected downstream.

**Evidence from user's log:**
```
[Phase 3 Debug] Model 0: AutoencodingEngine, unload_distorch_model=False
[UNLOAD_DEBUG] Adding to kept_models: AutoencodingEngine
[Phase 3 Debug] Model 2: FluxClipModel_, unload_distorch_model=False
[UNLOAD_DEBUG] Adding to kept_models: FluxClipModel_
[UNLOAD_DEBUG] Final counts - kept_models: 2, models_to_unload: 1
[UNLOAD_DEBUG] Updated mm.current_loaded_models, new count: 2
[DETECT_DEBUG] Checking DisTorch2 active status - loaded models: 0
```

**Suspicious:** After selective unload kept 2 models, detection shows "loaded models: 0"

**Possible causes:**
1. Object path mismatch between flag setting and reading
2. Flag not persisting through model operations
3. Detection logic reading from wrong location
4. Downstream cleanup (reset/GC/soft_empty) clearing retained models

## Phase 4 Viability Assessment

### Would Phase 4 Fix the Bug?

**Probably Not.** The bug appears to be:
- Flag storage/retrieval path mismatch, OR
- Retained models being cleared by downstream operations

Phase 4's WeakSet registry solves detection ambiguity, but doesn't address why retained models disappear.

### When Would Phase 4 Be Superior?

**If the bug is detection-related:** Phase 4's direct instance tracking eliminates all flag checking complexity.

**If the bug is downstream cleanup:** Phase 4 doesn't help - it adds MORE cleanup that could interfere.

### Hybrid Approach Consideration

**Option:** Keep Phase 3's selective unload, add Phase 4's registry for detection:

```python
# Use WeakSet for tracking but keep flag-based trigger
_kept_models_registry = weakref.WeakSet()

# In distorch nodes:
if keep_loaded:
    register_as_kept(out[0])

# In patched unload:
for lm in mm.current_loaded_models:
    if lm.model in _kept_models_registry:
        kept_models.append(lm)
    else:
        models_to_unload.append(lm)
```

## Recommendations

### Short-term (Debug Phase 3)
1. **Add comprehensive logging** to track object identity across flag set → flag read
2. **Verify flag persistence** through model operations
3. **Instrument post-unload flow** to detect where retained models disappear
4. **Git archaeology** to find when selective retention worked

### Long-term (If Phase 3 unfixable)
1. **Implement Phase 4** as proven alternative
2. **Remove Phase 3 patches** to avoid conflicts
3. **Keep minimal VRAM management** (model_memory_required patch)
4. **Extensive testing** for state consistency

## Conclusion

**Phase 4 is architecturally elegant** and eliminates detection ambiguity through direct instance tracking. However:

1. **Timing is not the issue** - Phase 3 already runs post-execution via deferred flags
2. **The bug is likely flag storage/detection** - which Phase 4 solves with WeakSet
3. **Risk of state conflicts** - direct manipulation of `mm.current_loaded_models` could break Comfy's tracking

**Recommended Path:**
1. Debug Phase 3 thoroughly with instrumentation (object identity tracking)
2. If root cause is flag detection → migrate to Phase 4's WeakSet registry
3. If root cause is downstream cleanup → Phase 4 won't help, need different solution

## Appendix: Code References

### Phase 3 Implementation Status
- **Flag setting:** `distorch_2.py` lines 475, 588, 694 (all three override classes)
- **Selective unload:** `model_management_mgpu.py` lines 140-180
- **Manager parity:** `model_management_mgpu.py` lines 100-130
- **Soft empty patch:** `__init__.py` lines 150-200
