# Code References (Definitive): ComfyUI Manager “Free model and node cache”

Purpose
- Provide an end-to-end, fully verified lineage of the ComfyUI Manager “Free model and node cache” button through to the exact consumption of flags in ComfyUI core, with exact file paths and code excerpts captured from the current snapshot in this workspace.

End‑to‑End Flow (Current Snapshot)
1) UI Button (Manager) → 2) JS helper free_models(...) → 3) POST /free (Comfy core) → 4) main.py prompt_worker thread polls flags and performs:
   - unload_models: comfy.model_management.unload_all_models()
   - free_memory: PromptExecutor.reset()
   - Additionally triggers GC and comfy.model_management.soft_empty_cache()

A) Frontend UI trigger (ComfyUI Manager)
- File: ../ComfyUI-Manager/js/comfyui-manager.js
- Location: app.registerExtension({ name: "Comfy.ManagerMenu", ... }) → setup() → ComfyButtonGroup
```js
new(await import("../../scripts/ui/components/button.js")).ComfyButton({
  icon: "vacuum-outline",
  action: () => {
    free_models();
  },
  tooltip: "Unload Models"
}).element,
new(await import("../../scripts/ui/components/button.js")).ComfyButton({
  icon: "vacuum",
  action: () => {
    free_models(true);
  },
  tooltip: "Free model and node cache"
}).element,
```
Semantics:
- “Unload Models” → free_models() (models only)
- “Free model and node cache” → free_models(true) (models + execution cache)

B) Frontend request construction (ComfyUI Manager)
- File: ../ComfyUI-Manager/js/common.js
- Function: export async function free_models(free_execution_cache)
```js
export async function free_models(free_execution_cache) {
  try {
    let mode = "";
    if (free_execution_cache) {
      mode = '{"unload_models": true, "free_memory": true}';
    } else {
      mode = '{"unload_models": true}';
    }

    console.log(`[ManagerFreePath] POST /free payload: ${mode}`);
    let res = await api.fetchApi(`/free`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: mode
    });
    console.log(`[ManagerFreePath] /free status: ${res.status}`);

    if (res.status == 200) {
      if (free_execution_cache) {
        showToast("'Models' and 'Execution Cache' have been cleared.", 3000);
      } else {
        showToast("Models' have been unloaded.", 3000);
      }
    } else {
      showToast('Unloading of models failed. Installed ComfyUI may be an outdated version.', 5000);
    }
  } catch (error) {
    console.error('[ManagerFreePath] /free error:', error);
    showToast('An error occurred while trying to unload models.', 5000);
  }
}
```
Semantics:
- free_models(true) → POST /free with {"unload_models": true, "free_memory": true}
- free_models() → POST /free with {"unload_models": true}

C) Core server endpoint (flags are set on the queue)
- File: ../../server.py
- Route: @routes.post("/free")
```py
@routes.post("/free")
async def post_free(request):
    json_data = await request.json()
    unload_models = json_data.get("unload_models", False)
    free_memory = json_data.get("free_memory", False)
    if unload_models:
        self.prompt_queue.set_flag("unload_models", unload_models)
    if free_memory:
        self.prompt_queue.set_flag("free_memory", free_memory)
    return web.Response(status=200)
```
Semantics:
- The HTTP endpoint itself does not unload/reset; instead it sets flags on PromptServer.prompt_queue for the background worker to consume.

D) Flag consumption and execution (definitive mechanism)
- File: ../../main.py
- Function: prompt_worker(q, server_instance)
- Excerpt (poll and handle flags, then clean up):
```py
        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()
```
Context:
- e is a PromptExecutor (created earlier in prompt_worker): `e = execution.PromptExecutor(server_instance, ...)`
- The worker thread is started in start_comfyui():
```py
threading.Thread(target=prompt_worker, daemon=True, args=(prompt_server.prompt_queue, prompt_server,)).start()
```

Interpretation (What the Manager button actually does)
- “Free model and node cache” sets unload_models: true and free_memory: true via POST /free.
- The background prompt_worker then:
  - Calls comfy.model_management.unload_all_models()
  - Calls e.reset() on the PromptExecutor to drop execution caches
  - Performs gc.collect() and comfy.model_management.soft_empty_cache()
- This matches the “benchmark button” behavior required for CPU memory reclamation (models fully unloaded + executor reset + allocator/cache cleanup).

Implications for MultiGPU P1 (force_full_system_cleanup)
- To 100% replicate the benchmark button behavior from within MultiGPU code paths:
  - Call comfy.model_management.unload_all_models()
  - Trigger PromptExecutor.reset() on the active executor
  - Follow up with gc.collect() and comfy.model_management.soft_empty_cache()
- Or, trigger the core behavior indirectly by POST /free with both flags set, relying on ComfyUI’s running prompt worker.

Verification Status
- All file paths and snippets above were extracted from this workspace:
  - Manager JS files under ../ComfyUI-Manager/js/
  - ComfyUI server and main under ../../server.py and ../../main.py
- Consumption site conclusively identified in ../../main.py prompt_worker via q.get_flags → unload_all_models + PromptExecutor.reset
