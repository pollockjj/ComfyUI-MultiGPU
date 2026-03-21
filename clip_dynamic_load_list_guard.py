import logging

import comfy.model_management
import comfy.model_patcher
from comfy.model_patcher import QuantizedTensor, get_key_weight, low_vram_patch_estimate_vram


logger = logging.getLogger("MultiGPU")

_PATCH_MARKER = "_mgpu_issue21_clip_dynamic_load_list_guard"
_MODULE_THRESHOLD = 200
_DEPTH_THRESHOLD = 200


def _iter_named_modules_nonrecursive(module):
    stack = [("", module)]
    seen = set()
    while stack:
        prefix, current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield prefix, current
        children = list(current._modules.items())
        for child_name, child in reversed(children):
            if child is None:
                continue
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            stack.append((child_prefix, child))


def _iter_named_parameters_nonrecursive(module):
    stack = [("", module)]
    seen = set()
    while stack:
        prefix, current = stack.pop()
        for name, param in current._parameters.items():
            if param is None:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        children = list(current._modules.items())
        for child_name, child in reversed(children):
            if child is None:
                continue
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            stack.append((child_prefix, child))


def _graph_requires_guard(module):
    stack = [(module, 0)]
    seen = set()
    module_count = 0
    max_depth = 0

    while stack:
        current, depth = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        module_count += 1
        max_depth = max(max_depth, depth)
        if module_count > _MODULE_THRESHOLD or max_depth > _DEPTH_THRESHOLD:
            return True
        for child in current._modules.values():
            if child is not None:
                stack.append((child, depth + 1))

    return False


def _safe_dynamic_load_list(self, default_device=None):
    loading = []
    for n, m in _iter_named_modules_nonrecursive(self.model):
        default = False
        params = dict(m.named_parameters(recurse=False))
        if params:
            for name, _ in _iter_named_parameters_nonrecursive(m):
                if name not in params:
                    default = True
                    break

        if default and default_device is not None:
            for param_name, param in params.items():
                param.data = param.data.to(
                    device=default_device,
                    dtype=getattr(m, param_name + "_comfy_model_dtype", None),
                )

        if not default and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
            module_mem = comfy.model_management.module_size(m)
            module_offload_mem = module_mem
            if hasattr(m, "comfy_cast_weights"):

                def check_module_offload_mem(key):
                    if key in self.patches:
                        return low_vram_patch_estimate_vram(self.model, key)
                    model_dtype = getattr(self.model, "manual_cast_dtype", None)
                    weight, _, _ = get_key_weight(self.model, key)
                    if model_dtype is None or weight is None:
                        return 0
                    if weight.dtype != model_dtype or isinstance(weight, QuantizedTensor):
                        return weight.numel() * model_dtype.itemsize
                    return 0

                module_offload_mem += check_module_offload_mem(f"{n}.weight")
                module_offload_mem += check_module_offload_mem(f"{n}.bias")

            sort_criteria = (module_offload_mem >= 64 * 1024, -module_offload_mem)
            loading.append(sort_criteria + (module_mem, n, m, params))

    return loading


def register_clip_dynamic_load_list_guard():
    original = comfy.model_patcher.ModelPatcherDynamic._load_list
    if getattr(original, _PATCH_MARKER, False):
        return False

    def guarded_load_list(self, for_dynamic=False, default_device=None):
        if not for_dynamic:
            return original(self, for_dynamic=for_dynamic, default_device=default_device)

        if _graph_requires_guard(self.model):
            logger.info("[MultiGPU Issue21] Using non-recursive ModelPatcherDynamic._load_list guard")
            return _safe_dynamic_load_list(self, default_device=default_device)

        return original(self, for_dynamic=for_dynamic, default_device=default_device)

    setattr(guarded_load_list, _PATCH_MARKER, True)
    comfy.model_patcher.ModelPatcherDynamic._load_list = guarded_load_list
    logger.info("[MultiGPU Issue21] Registered ModelPatcherDynamic._load_list guard")
    return True
