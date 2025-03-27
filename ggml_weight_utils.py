import torch
import importlib
import torch.cuda.nvtx as nvtx 
import logging
import comfy.model_management as mm
import comfy.model_management

GGMLTensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops').GGMLTensor
dequantize_tensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant').dequantize_tensor

from . import SMALL_TENSOR_THRESHOLD
COMPUTE_CACHE_LIMIT  = 0.66
TENSORATOR_CACHE_LIMIT  = 0.95
TENSORATOR_BUFFER_SIZE_MB  = 1024

patch_cache = {}

cached_tensor_map = {}
level_one_tensors = [] 
level_two_tensors = []
tensor_ring_buffer = []
ggml_ring_buffer = []
 
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:1")
compute_device = torch.device("cuda:0")
tensorator_device = torch.device("cuda:1")
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)
compute_target_cache = mm.get_total_memory(compute_device) * COMPUTE_CACHE_LIMIT / (1024 * 1024)
tensorator_target_cache = mm.get_total_memory(tensorator_device) * TENSORATOR_CACHE_LIMIT / (1024 * 1024)
compute_vram_used = mm.get_total_memory(compute_device)/ (1024 * 1024) - mm.get_free_memory(compute_device) / (1024 * 1024)
tensorator_vram_used = mm.get_total_memory(tensorator_device)/ (1024 * 1024) - mm.get_free_memory(tensorator_device) / (1024 * 1024)

if compute_device != tensorator_device:
    solo_gpu = False
else:
    solo_gpu = True
    tensorator_target_cache = 0

tensor_inference_order = 0

def move_patch_to_tensorator(item):
    stream = tensorator_stream
    if isinstance(item, torch.Tensor):
        if stream is not None:
            with torch.cuda.stream(stream):
                return item.to(tensorator_device, non_blocking=True)
        else:
            return item.to(tensorator_device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_tensorator(x) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_tensorator(x) for x in item]
    else:
        return item

def retrieve_cached_patch(patches_item, key):
    cache_key = tuple(key) if isinstance(key, (list, tuple)) else key
    if cache_key in patch_cache:
        return patch_cache[cache_key]
    patch = move_patch_to_tensorator(patches_item)
    patch_cache[cache_key] = patch
    return patch

def initialize_cache_levels():
    global cached_tensor_map
    
    total_tensor_size = sum(info['final_tensor_size'] for info in cached_tensor_map.values())
    threshold = total_tensor_size * SMALL_TENSOR_THRESHOLD
    
    all_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                    if info['cache_level'] == "uninitialized_cache_level"]

    all_tensors.sort(key=lambda x: (0 if x[1]['final_tensor_size'] < threshold else 1, 0 if x[1]['distorch_device'] == 'cpu' else 1,-x[1]['patch_qty'],-x[1]['final_tensor_size'], x[1]['tensor_inference_order']))

    for idx, (tensor_hash, _) in enumerate(all_tensors):
        if cached_tensor_map[tensor_hash]['final_tensor_size'] < threshold:
            cached_tensor_map[tensor_hash]['cache_priority'] = -1
            cached_tensor_map[tensor_hash]['cache_level'] = "prioritized" 
        else:
            cached_tensor_map[tensor_hash]['cache_priority'] = idx
            cached_tensor_map[tensor_hash]['cache_level'] = "prioritized"
        #print(f"Caching Assignment: cache_priority={cached_tensor_map[tensor_hash]['cache_priority']:4d} | name={cached_tensor_map[tensor_hash]['name']} | tensor_inference_order: {cached_tensor_map[tensor_hash]['tensor_inference_order']:3d} | patch_qty={cached_tensor_map[tensor_hash]['patch_qty']} | distorch_device={cached_tensor_map[tensor_hash]['distorch_device']} | size={cached_tensor_map[tensor_hash]['tensor_size']:.2f}MB | final_size={cached_tensor_map[tensor_hash]['final_tensor_size']:.2f}MB")

def initialize_ring_buffer():
    global cached_tensor_map, ggml_ring_buffer_size, ggml_ring_buffer_tensors, ggml_ring_buffer, tensor_ring_size

    ring_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                if info['cache_level'] == "uninitialized_ring_buffer"]

    ring_tensors.sort(key=lambda x: (x[1]['tensor_inference_order']))

    ggml_ring_buffer_size = 0
    ggml_ring_buffer_tensors = 0
    tensor_ring_size = 0
            
    for tensor, info in ring_tensors:
        if tensor_ring_size <= TENSORATOR_BUFFER_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "ggml_ring_buffer"
            tensor_ring_size += info['tensor_size']
            ggml_ring_buffer_size += 1
            cached_tensor_map[tensor]['buffer_index'] = ggml_ring_buffer_tensors
            ggml_ring_buffer_tensors += 1
        else:
            cached_tensor_map[tensor]['cache_level'] = "ggml_ring_buffer"
            cached_tensor_map[tensor]['buffer_index'] = ggml_ring_buffer_tensors
            ggml_ring_buffer_tensors += 1

    ggml_ring_buffer = []

    ggml_ring_buffer_index = sorted([t for t in ring_tensors if cached_tensor_map[t[0]]['cache_level'] in ["ggml_ring_buffer"]], key=lambda x: x[1]['tensor_inference_order'])   

    print(f"ggml_ring_buffer_tensors: {ggml_ring_buffer_tensors} | ggml_ring_buffer_size: {ggml_ring_buffer_size}")

    for i in range(ggml_ring_buffer_size):
        source_tensor_hash = ggml_ring_buffer_index[i][0] 
        prefetch_tensor = cached_tensor_map[source_tensor_hash]['source_tensor'].to(compute_device, non_blocking=True)
        ggml_ring_buffer.append(prefetch_tensor)

def cast_bias_weight_patched(s, input=None, dtype=None, device=None, bias_dtype=None):
    global tensor_inference_order
    from . import distorch_load_map
    
    if input is not None:
        if dtype is None:
            dtype = getattr(input, "dtype", torch.float32)
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = mm.device_supports_non_blocking(device)  # need to look futher into what this does
    if s.bias is not None:     
        source_tensor_hash_bias = s.bias.original_hash
        if source_tensor_hash_bias in distorch_load_map and distorch_load_map[source_tensor_hash_bias]['initialized'] is False:
            distorch_load_map[source_tensor_hash_bias]['tensor_inference_order'] = tensor_inference_order
            tensor_inference_order += 1
            distorch_load_map[source_tensor_hash_bias]['initialized'] = True
            bias = s.get_weight(s.bias.to(device), dtype, distorch_load_map[source_tensor_hash_bias]['tensor_inference_order'], distorch_load_map[source_tensor_hash_bias]['name'], distorch_load_map[source_tensor_hash_bias]['source_tensor'], distorch_load_map[source_tensor_hash_bias]['distorch_device'], source_tensor_hash_bias)
        elif source_tensor_hash_bias in distorch_load_map and distorch_load_map[source_tensor_hash_bias]['initialized'] is True:
            bias = s.get_weight(s.bias.to(device), dtype, distorch_load_map[source_tensor_hash_bias]['tensor_inference_order'], distorch_load_map[source_tensor_hash_bias]['name'], distorch_load_map[source_tensor_hash_bias]['source_tensor'], distorch_load_map[source_tensor_hash_bias]['distorch_device'], source_tensor_hash_bias)
        else:
            bias = s.get_weight(s.bias.to(device), dtype)
            logging.info(f"  Bias hash not found: 0x{source_tensor_hash_bias:x}")

        bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

    source_tensor_hash_weight = s.weight.original_hash

    if source_tensor_hash_weight in distorch_load_map and distorch_load_map[source_tensor_hash_weight]['initialized'] is False:
        distorch_load_map[source_tensor_hash_weight]['tensor_inference_order'] = tensor_inference_order
        tensor_inference_order += 1
        distorch_load_map[source_tensor_hash_weight]['initialized'] = True
        weight = s.get_weight(s.weight.to(device), dtype, distorch_load_map[source_tensor_hash_weight]['tensor_inference_order'], distorch_load_map[source_tensor_hash_weight]['name'], distorch_load_map[source_tensor_hash_weight]['source_tensor'], distorch_load_map[source_tensor_hash_weight]['distorch_device'], source_tensor_hash_weight)
    elif source_tensor_hash_weight in distorch_load_map and distorch_load_map[source_tensor_hash_weight]['initialized'] is True:
        weight = s.get_weight(s.weight.to(device), dtype, distorch_load_map[source_tensor_hash_weight]['tensor_inference_order'], distorch_load_map[source_tensor_hash_weight]['name'], distorch_load_map[source_tensor_hash_weight]['source_tensor'], distorch_load_map[source_tensor_hash_weight]['distorch_device'], source_tensor_hash_weight)
    else:
        weight = s.get_weight(s.weight.to(device), dtype)
    
    weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    
    return weight, bias

def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None, tensor_inference_order=None, name=None, source_tensor=None, distorch_device=None, source_tensor_hash=None):
    global cached_tensor_map, solo_gpu, level_one_tensors, level_two_tensors, compute_target_cache, tensorator_target_cache, compute_vram_used, tensorator_vram_used

    if source_tensor_hash in cached_tensor_map:
        if cached_tensor_map[source_tensor_hash]['cache_level'] == "level1":
            return cached_tensor_map[source_tensor_hash]['cached_final_tensor']
        elif cached_tensor_map[source_tensor_hash]['cache_level'] == "level2":
            with torch.cuda.stream(tensorator_stream):
                level_two_tensor = cached_tensor_map[source_tensor_hash]['cached_final_tensor']
                level_two_tensor.to(compute_device, non_blocking=True)
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor
        elif cached_tensor_map[source_tensor_hash]['cache_level'] == "uninitialized_cache_level":
            initialize_cache_levels()
        elif cached_tensor_map[source_tensor_hash]['cache_level'] == "uninitialized_ring_buffer":
            initialize_ring_buffer()

        patch_list = []
        for func, item, key in getattr(ggml_tensor, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches
                
        dequantized_tensor = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)

        if isinstance(dequantized_tensor, GGMLTensor):
            dequantized_tensor.__class__ = torch.Tensor

        if patch_list:
            if  patch_dtype is None:
                dequantized_tensor = func(patch_list, dequantized_tensor, key)
            else:
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                dequantized_tensor = func(patch_list, dequantized_tensor, key, patch_dtype)


        if source_tensor_hash in cached_tensor_map and cached_tensor_map[source_tensor_hash]['cache_level'] == "prioritized":  # SECOND PASS THROUGH - CACHE TENSORS VIA PRIORTIZATION SCHEME
            compute_vram_used = mm.get_total_memory(compute_device)/ (1024 * 1024) - mm.get_free_memory(compute_device) / (1024 * 1024)
            tensorator_vram_used = mm.get_total_memory(tensorator_device)/ (1024 * 1024) - mm.get_free_memory(tensorator_device) / (1024 * 1024)

            if compute_vram_used <= compute_target_cache or cached_tensor_map[source_tensor_hash]['cache_priority'] == -1:
                level_one_tensor = dequantized_tensor.clone().to(compute_device, non_blocking=True)
                level_one_tensor.original_hash = source_tensor_hash
                level_one_tensors.append(level_one_tensor)
                cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = level_one_tensor
                cached_tensor_map[source_tensor_hash]['cache_level'] = "level1"
                return level_one_tensor
            elif compute_vram_used > compute_target_cache:
                tensor_to_evict = None
                max_level1_priority_value = -1
                for cached_tensor in level_one_tensors:
                    if cached_tensor_map[cached_tensor.original_hash]['cache_priority'] > max_level1_priority_value:
                        max_level1_priority_value = cached_tensor_map[cached_tensor.original_hash]['cache_priority']
                        tensor_to_evict = cached_tensor
                if max_level1_priority_value > cached_tensor_map[source_tensor_hash]['cache_priority']:
                    with torch.cuda.stream(tensorator_stream): 
                        eviction_candidate_index = next(index for index, d in enumerate(level_one_tensors) if d is tensor_to_evict)
                        evicted_tensor_ref = level_one_tensors.pop(eviction_candidate_index)
                        if tensorator_vram_used < tensorator_target_cache:
                            xfered_to_level2 = evicted_tensor_ref.to(tensorator_device, non_blocking=True)
                            level_two_tensors.append(xfered_to_level2)
                            cached_tensor_map[tensor_to_evict.original_hash]['cache_level'] = "level2"
                            cached_tensor_map[tensor_to_evict.original_hash]['cached_final_tensor'] = xfered_to_level2
                        else:
                            cached_tensor_map[tensor_to_evict.original_hash]['cache_level'] = "uninitialized_ring_buffer"
                            cached_tensor_map[tensor_to_evict.original_hash]['cached_final_tensor'] = None
                        level_one_tensor = dequantized_tensor.clone().to(compute_device, non_blocking=True)
                        level_one_tensor.original_hash = source_tensor_hash
                        level_one_tensors.append(level_one_tensor)
                        cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = level_one_tensor
                        cached_tensor_map[source_tensor_hash]['cache_level'] = "level1"
                        tensorator_event.record(tensorator_stream)
                        torch.cuda.current_stream().wait_event(tensorator_event)
                    return level_one_tensor
                else:
                    if tensorator_vram_used < tensorator_target_cache:
                        with torch.cuda.stream(tensorator_stream):
                            level_two_tensor = dequantized_tensor.clone().to(tensorator_device, non_blocking=True)
                            level_two_tensor.original_hash = source_tensor_hash
                            level_two_tensors.append(level_two_tensor)
                            cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = level_two_tensor
                            cached_tensor_map[source_tensor_hash]['cache_level'] = "level2"
                            tensorator_event.record(tensorator_stream)
                            tensor_on_compute = level_two_tensor.to(compute_device, non_blocking=True)
                            tensorator_event.record(tensorator_stream)
                            torch.cuda.current_stream().wait_event(tensorator_event)
                        return tensor_on_compute
                    else:
                        cached_tensor_map[source_tensor_hash]['cache_level'] = "uninitialized_ring_buffer"
                        tensor_on_compute = dequantized_tensor.to(device=compute_device, non_blocking=True)
                        return tensor_on_compute
        elif source_tensor_hash in cached_tensor_map and cached_tensor_map[source_tensor_hash]['cache_level'] == "uninitialized_ring_buffer":
            ggml_ring_buffer.pop(0)
            ggml_current_index = cached_tensor_map[source_tensor_hash]['buffer_index'] 
            ggml_prefetch_position = ((ggml_current_index + tensor_ring_buffer_size) % total_buffered_tensors ) # identify the position of the tensor in the tensor ring buffer
            ggml_prefetch_tensor_hash = ggml_ring_buffer[ggml_prefetch_position]['original_hash']
            ggml_prefetch_tensor = cached_tensor_map[ggml_prefetch_tensor_hash]['source_tensor'].to(tensorator_device, non_blocking=True)
            ggml_ring_buffer.append(ggml_prefetch_tensor)
            

        dequantized_tensor = dequantized_tensor.to(device=compute_device, non_blocking=True)
    else:                                                                                                                                #First time through for a tensor
        patch_list = []
        for func, item, key in getattr(ggml_tensor, "patches", []):
            patches = retrieve_cached_patch(item, key)
            patch_list += patches
                
        dequantized_tensor = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)

        if isinstance(dequantized_tensor, GGMLTensor):
            dequantized_tensor.__class__ = torch.Tensor

        if patch_list:
            if  patch_dtype is None:
                dequantized_tensor = func(patch_list, dequantized_tensor, key)
            else:
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                dequantized_tensor = func(patch_list, dequantized_tensor, key, patch_dtype)
        
        if source_tensor_hash not in cached_tensor_map:  #VERY FIRST PASS THROUGH - ENUMERATE ALL TENSORS IN INFERENCE ORDER
            cached_tensor_map[source_tensor_hash] = {}
            cached_tensor_map[source_tensor_hash]['tensor_inference_order'] = tensor_inference_order
            cached_tensor_map[source_tensor_hash]['patch_qty'] = len(patch_list)
            cached_tensor_map[source_tensor_hash]['tensor_size'] = (ggml_tensor.numel() * ggml_tensor.element_size() / (1024 * 1024))
            cached_tensor_map[source_tensor_hash]['distorch_device'] = distorch_device
            cached_tensor_map[source_tensor_hash]['cache_level'] = "uninitialized_cache_level"
            cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = None
            cached_tensor_map[source_tensor_hash]['name'] = name
            cached_tensor_map[source_tensor_hash]['dtype'] = dtype
            cached_tensor_map[source_tensor_hash]['dequant_dtype'] = dequant_dtype
            cached_tensor_map[source_tensor_hash]['patches'] = ggml_tensor.patches
            cached_tensor_map[source_tensor_hash]['patch_dtype'] = patch_dtype
            cached_tensor_map[source_tensor_hash]['source_tensor'] = source_tensor
            cached_tensor_map[source_tensor_hash]['final_tensor_size'] = dequantized_tensor.numel() * dequantized_tensor.element_size() / (1024 * 1024)
            cached_tensor_map[source_tensor_hash]['cache_priority'] = -1
            #print(f"Initializing Cache Entry: hash=0x{source_tensor_hash:x} | tensor_inference_order={tensor_inference_order:3d} | patches={len(patch_list)} | device={distorch_device} | size={cached_tensor_map[source_tensor_hash]['tensor_size']:.2f}MB | name={name}")

    return dequantized_tensor