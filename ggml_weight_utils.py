import torch
import importlib
import torch.cuda.nvtx as nvtx 
import gguf
import comfy.model_management

GGMLTensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops').GGMLTensor
dequantize_tensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant').dequantize_tensor

COMPUTE_CACHE_SIZE_MB  = 1024
TENSORATOR_CACHE_SIZE_MB  = 4096
TENSORATOR_BUFFER_SIZE_MB  = 1024

patch_cache = {}

cached_tensor_map = {}
level_one_tensors = [] 
level_two_tensors = []
tensor_ring_buffer = []
ggml_ring_buffer = []
 
# hard-coded streams and variables for compute and tensorator during development
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:0")
compute_device = torch.device("cuda:0")
tensorator_device = torch.device("cuda:0")

# Setup events for cross-device synchronization
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)

# Global variable for tracking inference order
cast_bias_weight_inf_ord = 0

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
    global cached_tensor_map, tensor_ring_buffer, solo_gpu

    all_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                    if info['cache_level'] == "uninitialized"]

    all_tensors.sort(key=lambda x: (-x[1]['patch_qty'], x[1]['tensor_size']))

    cumulative_size = 0
    level1_size = 0
    level2_size = 0
    tensor_ring_size = 0
    tensor_ring_count = 0
    
    for tensor, info in all_tensors:
        cumulative_size += info['tensor_size']
        if level1_size <= COMPUTE_CACHE_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "level1"
            level1_size += info['tensor_size']
        elif level2_size <= TENSORATOR_CACHE_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "level2"
            level2_size += info['tensor_size']
        elif tensor_ring_size <= TENSORATOR_BUFFER_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "tensor_ring"
            tensor_ring_size += info['tensor_size']
            tensor_ring_count += 1
        else:
            cached_tensor_map[tensor]['cache_level'] = "tensor_ring"

    if compute_device != tensorator_device:
        tensor_ring_count = tensor_ring_count // 2
        solo_gpu = False
    else:
        solo_gpu = True
       
    tensor_ring_buffer_index = sorted([t for t in all_tensors if cached_tensor_map[t[0]]['cache_level'] in ["tensor_ring"]], key=lambda x: x[1]['index'])   
    tensor_ring_buffer = []
    ggml_ring_buffer = []
    with torch.cuda.stream(tensorator_stream): 
        for i in range(min(tensor_ring_count, len(tensor_ring_buffer_index))):

            tensor_hash = tensor_ring_buffer_index[i][0]
            prefetch_tensor = cached_tensor_map[tensor_hash].to(tensorator_device, non_blocking=True)
            ggml_ring_buffer.append(prefetch_tensor)
            
            if solo_gpu == False:
                patch_list = []
                for func, item, key in getattr(prefetch_tensor, cached_tensor_map[tensor_hash]['patches'], []):
                    patches = retrieve_cached_patch(item, key)
                    patch_list += patches
            
                prefetch_tensor = dequantize_tensor(prefetch_tensor, cached_tensor_map[tensor_hash]['dtype'], cached_tensor_map[tensor_hash]['dequant_dtype'])
            
                if GGMLTensor is not None and isinstance(prefetch_tensor, GGMLTensor):
                    prefetch_tensor.__class__ = torch.Tensor
                
                if patch_list:
                    if cached_tensor_map[tensor_hash]['patch_dtype'] is None:
                        prefetch_tensor = func(patch_list, prefetch_tensor, key)
                    else:
                        prefetch_tensor = func(patch_list, prefetch_tensor, key, cached_tensor_map[tensor_hash]['dtype'] if cached_tensor_map[tensor_hash]['patch_dtype']=="target" else cached_tensor_map[tensor_hash]['patch_dtype'])

                prefetch_tensor = cached_tensor_map[tensor_hash]['tensor'].to(tensorator_device, non_blocking=True)
                tensor_ring_buffer.append((tensor_hash, prefetch_tensor))
    torch.cuda.current_stream().wait_event(tensorator_event)

@profile
def cast_bias_weight_patched(s, input=None, dtype=None, device=None, bias_dtype=None):
    global cast_bias_weight_inf_ord
    from . import distorch_load_map
    
    if input is not None:
        if dtype is None:
            dtype = getattr(input, "dtype", torch.float32)
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    ggml_tensor_hash = s.weight.original_hash

    if ggml_tensor_hash in distorch_load_map and distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] is False:
        distorch_load_map[ggml_tensor_hash]['inf_order'] = cast_bias_weight_inf_ord
        cast_bias_weight_inf_ord += 1
        distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] = True

    bias = None
    non_blocking = comfy.model_management.device_supports_non_blocking(device)
    if s.bias is not None:
        bias = s.get_weight(s.bias.to(device), dtype)
        bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

    kwargs = {}
    if ggml_tensor_hash in distorch_load_map and 'inf_order' in distorch_load_map[ggml_tensor_hash]:
        kwargs['index'] = distorch_load_map[ggml_tensor_hash]['inf_order']
        kwargs['name'] = distorch_load_map[ggml_tensor_hash]['name']
        kwargs['distorch_device'] = distorch_load_map[ggml_tensor_hash]['distorch_device']
    kwargs['ggml_tensor_hash'] = ggml_tensor_hash
    
    try:
        weight = s.get_weight(s.weight.to(device), dtype, **kwargs)
    except TypeError:
        weight = s.get_weight(s.weight.to(device), dtype)
    weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    
    return weight, bias
@profile
def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None, index=None, name=None, ggml_tensor_hash=None, distorch_device=None):
    global cached_tensor_map, solo_gpu, tensor_ring_buffer, level_one_tensors, level_two_tensors
    if ggml_tensor is None:
        return None

    with torch.cuda.stream(tensorator_stream):   

        if ggml_tensor_hash in cached_tensor_map:
            if cached_tensor_map[ggml_tensor_hash]['cache_level'] == "level1" and cached_tensor_map[ggml_tensor_hash]['cached_tensor'] is not None:                # Level 1 cache
                return cached_tensor_map[ggml_tensor_hash]['cached_tensor']
            elif cached_tensor_map[ggml_tensor_hash]['cache_level'] == "level2" and cached_tensor_map[ggml_tensor_hash]['cached_tensor'] is not None:              # Level 2 cache
                with torch.cuda.stream(tensorator_stream):
                    level_two_tensor = cached_tensor_map[ggml_tensor_hash]['cached_tensor']
                    level_two_tensor.to(compute_device, non_blocking=True)
                    tensorator_event.record(tensorator_stream)
                    torch.cuda.current_stream().wait_event(tensorator_event)
                    return level_two_tensor
            elif cached_tensor_map[ggml_tensor_hash]['cache_level'] == "uninitialized":
                initialize_cache_levels()

        if solo_gpu == True:
            
            patch_list = []
            for func, item, key in getattr(ggml_tensor, "patches", []):
                patches = retrieve_cached_patch(item, key)
                patch_list += patches

            tensorator_tensor = dequantize_tensor(ggml_ring_buffer.pop(0), dtype, dequant_dtype)
            
            if GGMLTensor is not None and isinstance(tensorator_tensor, GGMLTensor):
                tensorator_tensor.__class__ = torch.Tensor
            if patch_list:
                if patch_dtype is None:
                    tensorator_tensor = func(patch_list, tensorator_tensor, key)
                else:
                    tensorator_tensor = func(patch_list, tensorator_tensor, key, dtype if patch_dtype=="target" else patch_dtype)

            ggml_current_index = next((idx for idx, (hash_val, _) in enumerate(ggml_ring_buffer) if hash_val == ggml_tensor_hash), 0) # identify the index of the tensor in the tensor ring buffer
            ggml_prefetch_position = (ggml_current_index) % len(ggml_ring_buffer) # identify the position of the tensor in the tensor ring buffer
            ggml_prefetch_tensor_hash = ggml_ring_buffer[ggml_prefetch_position][0]
            ggml_prefetch_tensor = cached_tensor_map[ggml_prefetch_tensor_hash].to(tensorator_device, non_blocking=True)
            ggml_ring_buffer.append((ggml_prefetch_tensor_hash, ggml_prefetch_tensor))
        else:
            ggml_ring_buffer.pop(0)
            ggml_current_index = next((idx for idx, (hash_val, _) in enumerate(ggml_ring_buffer) if hash_val == ggml_tensor_hash), 0) # identify the index of the tensor in the tensor ring buffer
            ggml_prefetch_position = (ggml_current_index) % len(ggml_ring_buffer) # identify the position of the tensor in the tensor ring buffer
            ggml_prefetch_tensor_hash = ggml_ring_buffer[ggml_prefetch_position][0]
            ggml_prefetch_tensor = cached_tensor_map[ggml_prefetch_tensor_hash].to(tensorator_device, non_blocking=True)
            ggml_ring_buffer.append((ggml_prefetch_tensor_hash, ggml_prefetch_tensor))
            
            patch_list = []
            for func, item, key in getattr(ggml_prefetch_tensor, cached_tensor_map[ggml_prefetch_tensor_hash]['patches'], []):
                patches = retrieve_cached_patch(item, key)
                patch_list += patches
            
            prefetch_tensor = dequantize_tensor(ggml_prefetch_tensor, cached_tensor_map[ggml_prefetch_tensor_hash]['dtype'], cached_tensor_map[ggml_prefetch_tensor_hash]['dequant_dtype'])
        
            if GGMLTensor is not None and isinstance(prefetch_tensor, GGMLTensor):
                prefetch_tensor.__class__ = torch.Tensor
            
            if patch_list:
                if cached_tensor_map[ggml_prefetch_tensor_hash]['patch_dtype'] is None:
                    prefetch_tensor = func(patch_list, prefetch_tensor, key)
                else:
                    prefetch_tensor = func(patch_list, prefetch_tensor, key, cached_tensor_map[ggml_prefetch_tensor_hash]['dtype'] if cached_tensor_map[ggml_prefetch_tensor_hash]['patch_dtype']=="target" else cached_tensor_map[ggml_prefetch_tensor_hash]['patch_dtype'])

            prefetch_tensor = cached_tensor_map[ggml_prefetch_tensor_hash]['tensor'].to(tensorator_device, non_blocking=True)
            tensor_ring_buffer.append((ggml_prefetch_tensor_hash, prefetch_tensor))
            
            tensorator_tensor = tensor_ring_buffer.pop(0)
        

        if ggml_tensor_hash in cached_tensor_map and cached_tensor_map[ggml_tensor_hash]['cache_level'] == "level1":                #second time through for a level1-assigned tensor as level 1 branches after the first time
            level_one_tensor = tensorator_tensor.clone().to(compute_device, non_blocking=True)
            level_one_tensors.append(level_one_tensor)
            cached_tensor_map[ggml_tensor_hash]['cached_tensor'] = level_one_tensor
            # print(f"Moving Dequantized and Patched Tensor: 0x{ggml_tensor_hash:x} | Index: {cached_tensor_map[ggml_tensor_hash]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_hash]['tensor_size']:.2f} | to compute_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_one_tensor
        elif ggml_tensor_hash in cached_tensor_map and cached_tensor_map[ggml_tensor_hash]['cache_level'] == "level2":
            level_two_tensor = tensorator_tensor.clone().to(tensorator_device, non_blocking=True)
            level_two_tensors.append(level_two_tensor)
            cached_tensor_map[ggml_tensor_hash]['cached_tensor'] = level_two_tensor
            # print(f"Moving Dequantized and Patched Tensor: 0x{ggml_tensor_hash:x} | Index: {cached_tensor_map[ggml_tensor_hash]['index']:3d} | Size: {cached_tensor_map[ggml_tensor_hash]['tensor_size']:.2f} | to tensorator_device")
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return level_two_tensor
        elif ggml_tensor_hash in cached_tensor_map and cached_tensor_map[ggml_tensor_hash]['cache_level'] == "tensor_ring":          #Assumes second video card is tensorator_device, should create a full-tensor ring buffer for compute_device 
            current_index = next((idx for idx, (hash_val, _) in enumerate(tensor_ring_buffer) if hash_val == ggml_tensor_hash), 0)     
            prefetch_position = (current_index) % len(tensor_ring_buffer)
            prefetch_tensor_hash = tensor_ring_buffer[prefetch_position][0]
            prefetch_tensor = cached_tensor_map[prefetch_tensor_hash]['tensor'].to(tensorator_device, non_blocking=True)
            tensor_ring_buffer.append((prefetch_tensor_hash, prefetch_tensor))
            buffered_tensor = tensor_ring_buffer.pop(0)
            tensorator_event.record(tensorator_stream)
            torch.cuda.current_stream().wait_event(tensorator_event)
            return buffered_tensor

        tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
        tensorator_event.record(tensorator_stream)
            
    if ggml_tensor_hash not in cached_tensor_map and ggml_tensor_hash is not None:
        cached_tensor_map[ggml_tensor_hash] = {}
        cached_tensor_map[ggml_tensor_hash]['index'] = index
        cached_tensor_map[ggml_tensor_hash]['patch_qty'] = len(patch_list)
        cached_tensor_map[ggml_tensor_hash]['tensor_size'] = (ggml_tensor.numel() * ggml_tensor.element_size() / (1024 * 1024))
        cached_tensor_map[ggml_tensor_hash]['distorch_device'] = distorch_device
        cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized"
        cached_tensor_map[ggml_tensor_hash]['cached_tensor'] = None
        cached_tensor_map[ggml_tensor_hash]['name'] = name
        cached_tensor_map[ggml_tensor_hash]['dtype'] = dtype
        cached_tensor_map[ggml_tensor_hash]['dequant_dtype'] = dequant_dtype
        cached_tensor_map[ggml_tensor_hash]['patches'] = ggml_tensor.patches
        cached_tensor_map[ggml_tensor_hash]['patch_dtype'] = patch_dtype
        print(f"Caching Initialization: ptr=0x{ggml_tensor_hash:x} | index={index} | patches={len(patch_list)} | device={distorch_device} | size={cached_tensor_map[ggml_tensor_hash]['tensor_size']:.2f}MB | name={name} ")
  
    torch.cuda.current_stream().wait_event(tensorator_event)
    return tensorator_tensor