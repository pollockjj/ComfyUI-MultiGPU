No. It is clear that you do not given multiple failed implementations past this point. So, lets do this in phases. 

Phase 1: Implement DISTORCH2_UNLOAD_MODEL Global correctly. It should be set to True when it sees a keep_loaded=false and should be reset at the end of our patched unload_all_models. No other code changes. Document with device snapshot and memory datalog each time a new operation is done - so when it is set and unset so it can been seen in the datalog.

Phase 2: In Distorch_2.py, implement `_mgpu_unload` flag to any DisTorch model when keep_loaded=false and at the same time as setting DISTORCH2_UNLOAD_MODEL=True. In our patched unload_all_models() we create a simple evaluatioon loop with my pseudocode:

if hasattr(getattr(model, 'model', None), '_mgpu_unload'):
    multigpu_memory_log(model_hash, "_mgpu_unload=true")
    logger.mgpu_mm_log(f"[THREE_FLAG_DEBUG] model has `_mpgu_unload` flag")
else:
    logger.mgpu_mm_log(f"[THREE_FLAG_DEBUG] model does not have _mpgu_unload flag")

At the end of the loop no matter what calls it, DISTORCH2_UNLOAD_MODEL = FALSE with an appropriate log:
    logger.mgpu_mm_log("Setting DISTORCH2_UNLOAD_MODEL=False")

Phase 3: Replace existing faulty retention or ejection logic with the loop from Phase 2:

1. At the beginning of our patched unload_all_models, check DISTORCH2_UNLOAD_MODEL
    If FALSE: run _original_unload_all_models()
    IF  TRUE: Using the loop from Phase 2, apply only the unload_all_models routine to the models with `_mpgu_unload` flag set, else do nothing to other models, exactly like Else loop from Phase 2.  
