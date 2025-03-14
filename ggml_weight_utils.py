import torch
import importlib
import torch.cuda.nvtx as nvtx 
import logging
import comfy.model_management

GGMLTensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.ops').GGMLTensor
dequantize_tensor = importlib.import_module('custom_nodes.ComfyUI-GGUF.dequant').dequantize_tensor

from . import SMALL_TENSOR_THRESHOLD
COMPUTE_CACHE_SIZE_MB  = 4096
TENSORATOR_CACHE_SIZE_MB  = 8192
TENSORATOR_BUFFER_SIZE_MB  = 0
DISABLE_RING_BUFFER = True

patch_cache = {}

cached_tensor_map = {}
level_one_tensors = [] 
level_two_tensors = []
tensor_ring_buffer = []
ggml_ring_buffer = []
 
# hard-coded streams and variables for compute and tensorator during development
compute_stream = torch.cuda.Stream(device="cuda:0") 
tensorator_stream = torch.cuda.Stream(device="cuda:1")
compute_device = torch.device("cuda:0")
tensorator_device = torch.device("cuda:1")

# Setup events for cross-device synchronization
compute_event = torch.cuda.Event(enable_timing=False)
tensorator_event = torch.cuda.Event(enable_timing=False)

# Global variable for tracking inference order
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
    global cached_tensor_map, tensor_ring_buffer, ggml_ring_buffer, solo_gpu
    
    total_tensor_size = sum(info['final_tensor_size'] for info in cached_tensor_map.values())
    threshold = total_tensor_size * SMALL_TENSOR_THRESHOLD
    
    all_tensors = [(tensor, info) for tensor, info in cached_tensor_map.items()
                    if info['cache_level'] == "uninitialized"]

    all_tensors.sort(key=lambda x: (-x[1]['patch_qty'], x[1]['final_tensor_size'], 0 if x[1]['distorch_device'] == 'cpu' else 1))

    cumulative_size = 0
    level1_size = 0
    level2_size = 0
    tensor_ring_size = 0
    tensor_ring_count = 0
    
    for tensor, info in all_tensors:
        cumulative_size += info['final_tensor_size']
        if level1_size <= COMPUTE_CACHE_SIZE_MB or info['final_tensor_size'] < threshold:
            cached_tensor_map[tensor]['cache_level'] = "level1"
            level1_size += info['final_tensor_size']
        elif level2_size <= TENSORATOR_CACHE_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "level2"
            level2_size += info['final_tensor_size']
        elif tensor_ring_size <= TENSORATOR_BUFFER_SIZE_MB:
            cached_tensor_map[tensor]['cache_level'] = "tensor_ring"
            tensor_ring_size += info['tensor_size']
            tensor_ring_count += 1
        else:
            cached_tensor_map[tensor]['cache_level'] = "tensor_ring"
        print(f"Caching Assignment: name={cached_tensor_map[tensor]['name']} | patch_qty={cached_tensor_map[tensor]['patch_qty']} | distorch_device={cached_tensor_map[tensor]['distorch_device']} | size={cached_tensor_map[tensor]['tensor_size']:.2f}MB | final_size={cached_tensor_map[tensor]['final_tensor_size']:.2f}MB | cache_level={cached_tensor_map[tensor]['cache_level']}")
    summary = {}
    for tensor, info in cached_tensor_map.items():
        device = info.get('distorch_device', 'unknown')
        level = info.get('cache_level', 'uninitialized')
        if device not in summary:
            summary[device] = {}
        if level not in summary[device]:
            summary[device][level] = {'count': 0, 'total_size': 0.0}
        summary[device][level]['count'] += 1
        summary[device][level]['total_size'] += info.get('tensor_size', 0.0)
    
    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<12}{:>10}{:>14}{:>14}"
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(eq_line)
    logging.info(" Cache Summary by Device and Cache Level")
    logging.info(eq_line)
    logging.info(fmt_assign.format("Device", "Level", "Count", "Total (MB)"))
    logging.info(dash_line)
    
    order = {"level1": 0, "level2": 1, "tensor_ring": 2, "uninitialized": 3}
    for dev in sorted(summary.keys()):
        for level in sorted(summary[dev].keys(), key=lambda lvl: order.get(lvl, 99)):
            count = summary[dev][level]['count']
            total_mb = summary[dev][level]['total_size']
            logging.info(fmt_assign.format(dev, level, count, f"{total_mb:.2f}"))
    logging.info(dash_line)

    if compute_device != tensorator_device:
        tensor_ring_count = tensor_ring_count // 2
        solo_gpu = False
    else:
        solo_gpu = True
       
    tensor_ring_buffer_index = sorted([t for t in all_tensors if cached_tensor_map[t[0]]['cache_level'] in ["tensor_ring"]], key=lambda x: x[1]['tensor_inference_order'])   
    tensor_ring_buffer = []
    ggml_ring_buffer = []
    with torch.cuda.stream(tensorator_stream): 
        for i in range(min(tensor_ring_count, len(tensor_ring_buffer_index))):
            
            source_tensor_hash = tensor_ring_buffer_index[i][0] 
            prefetch_tensor = cached_tensor_map[source_tensor_hash]['source_tensor'].to(tensorator_device, non_blocking=True)
            ggml_ring_buffer.append(prefetch_tensor)
            
            if solo_gpu == False:
                patch_list = []
                # Access patches directly from cached map instead of trying to get them as an attribute
                for func, item, key in cached_tensor_map[source_tensor_hash]['patches']:
                    patches = retrieve_cached_patch(item, key)
                    patch_list += patches
            
                prefetch_tensor = dequantize_tensor(prefetch_tensor, cached_tensor_map[source_tensor_hash]['dtype'], cached_tensor_map[source_tensor_hash]['dequant_dtype'])
            
                if GGMLTensor is not None and isinstance(prefetch_tensor, GGMLTensor):
                    prefetch_tensor.__class__ = torch.Tensor
                
                if patch_list:
                    if cached_tensor_map[source_tensor_hash]['patch_dtype'] is None:
                        prefetch_tensor = func(patch_list, prefetch_tensor, key)
                    else:
                        prefetch_tensor = func(patch_list, prefetch_tensor, key, cached_tensor_map[source_tensor_hash]['dtype'] if cached_tensor_map[source_tensor_hash]['patch_dtype']=="target" else cached_tensor_map[source_tensor_hash]['patch_dtype'])

                tensor_ring_buffer.append(prefetch_tensor)
    torch.cuda.current_stream().wait_event(tensorator_event)

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
    non_blocking = comfy.model_management.device_supports_non_blocking(device)  # need to look futher into what this does
    if s.bias is not None:     
        source_tensor_hash_bias = s.bias.original_hash
        if source_tensor_hash_bias in distorch_load_map and distorch_load_map[source_tensor_hash_bias]['initialized'] is False:
            distorch_load_map[source_tensor_hash_bias]['tensor_inference_order'] = tensor_inference_order
            tensor_inference_order += 1
            distorch_load_map[source_tensor_hash_bias]['initialized'] = True
            bias = s.get_weight(s.bias.to(device), dtype, distorch_load_map[source_tensor_hash_bias]['tensor_inference_order'], distorch_load_map[source_tensor_hash_bias]['name'], distorch_load_map[source_tensor_hash_bias]['source_tensor'], distorch_load_map[source_tensor_hash_bias]['distorch_device'], source_tensor_hash_bias)
            #logging.info(f"  Bias hash found: 0x{source_tensor_hash_bias:x} | tensor_inference_order: {distorch_load_map[source_tensor_hash_bias]['tensor_inference_order']:3d} | Size: {distorch_load_map[source_tensor_hash_bias]['tensor_size']:3.2f}MB | To compute: {device} | name: {distorch_load_map[source_tensor_hash_bias]['name']}")          
        elif source_tensor_hash_bias in distorch_load_map and distorch_load_map[source_tensor_hash_bias]['initialized'] is True:
            bias = s.get_weight(s.bias.to(device), dtype, distorch_load_map[source_tensor_hash_bias]['tensor_inference_order'], distorch_load_map[source_tensor_hash_bias]['name'], distorch_load_map[source_tensor_hash_bias]['source_tensor'], distorch_load_map[source_tensor_hash_bias]['distorch_device'], source_tensor_hash_bias)
            #logging.info(f"  Bias hash hit: 0x{source_tensor_hash_bias:x} | tensor_inference_order: {distorch_load_map[source_tensor_hash_bias]['tensor_inference_order']:3d} | Size: {distorch_load_map[source_tensor_hash_bias]['tensor_size']:3.2f}MB | To compute: {device} | name: {distorch_load_map[source_tensor_hash_bias]['name']}")
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
        #logging.info(f"Weight hash found: 0x{source_tensor_hash_weight:x} | tensor_inference_order: {distorch_load_map[source_tensor_hash_weight]['tensor_inference_order']:3d} | Size: {distorch_load_map[source_tensor_hash_weight]['tensor_size']:3.2f}MB | To compute: {device} | name: {distorch_load_map[source_tensor_hash_weight]['name']}" )     
    elif source_tensor_hash_weight in distorch_load_map and distorch_load_map[source_tensor_hash_weight]['initialized'] is True:
        weight = s.get_weight(s.weight.to(device), dtype, distorch_load_map[source_tensor_hash_weight]['tensor_inference_order'], distorch_load_map[source_tensor_hash_weight]['name'], distorch_load_map[source_tensor_hash_weight]['source_tensor'], distorch_load_map[source_tensor_hash_weight]['distorch_device'], source_tensor_hash_weight)
        #logging.info(f"Weight hash hit: 0x{source_tensor_hash_weight:x} | tensor_inference_order: {distorch_load_map[source_tensor_hash_weight]['tensor_inference_order']:3d} | Size: {distorch_load_map[source_tensor_hash_weight]['tensor_size']:3.2f}MB | To compute: {device} | name: {distorch_load_map[source_tensor_hash_weight]['name']}" )
    else:
        weight = s.get_weight(s.weight.to(device), dtype)
        #logging.info(f"  Weight hash not found: 0x{source_tensor_hash_weight:x}")
 

    weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    
    return weight, bias
@profile
def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None, tensor_inference_order=None, name=None, source_tensor=None, distorch_device=None, source_tensor_hash=None):
    global cached_tensor_map, solo_gpu, tensor_ring_buffer, level_one_tensors, level_two_tensors
    if ggml_tensor is None:
        return None

    with torch.cuda.stream(tensorator_stream):   

        if source_tensor_hash in cached_tensor_map:
            if cached_tensor_map[source_tensor_hash]['cache_level'] == "level1" and cached_tensor_map[source_tensor_hash]['cached_final_tensor'] is not None:                # Level 1 cache
                #print(f"Level 1 Cache Hit: name={cached_tensor_map[source_tensor_hash]['name']} | size={cached_tensor_map[source_tensor_hash]['tensor_size']:.2f}MB | cache_level={cached_tensor_map[source_tensor_hash]['cache_level']}")
                return cached_tensor_map[source_tensor_hash]['cached_final_tensor']
            elif cached_tensor_map[source_tensor_hash]['cache_level'] == "level2" and cached_tensor_map[source_tensor_hash]['cached_final_tensor'] is not None:              # Level 2 cache
                with torch.cuda.stream(tensorator_stream):
                    level_two_tensor = cached_tensor_map[source_tensor_hash]['cached_final_tensor']
                    level_two_tensor.to(compute_device, non_blocking=True)
                    tensorator_event.record(tensorator_stream)
                    torch.cuda.current_stream().wait_event(tensorator_event)
                    return level_two_tensor
            elif cached_tensor_map[source_tensor_hash]['cache_level'] == "uninitialized":
                initialize_cache_levels()

            if DISABLE_RING_BUFFER:
                if ggml_tensor is None:
                    return

                patch_list = []
                for func, item, key in getattr(ggml_tensor, "patches", []):
                    patches = retrieve_cached_patch(item, key)
                    patch_list += patches
                        
                tensorator_tensor = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)

                if isinstance(tensorator_tensor, GGMLTensor):
                    tensorator_tensor.__class__ = torch.Tensor

                if patch_list:
                    if  patch_dtype is None:
                        tensorator_tensor = func(patch_list, tensorator_tensor, key)
                    else:
                        patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                        tensorator_tensor = func(patch_list, tensorator_tensor, key, patch_dtype)


            elif solo_gpu == True:
                
                source_hash = getattr(source_tensor, "original_hash", None)
                ggml_current_index = next( (idx for idx, tensor in enumerate(ggml_ring_buffer) if getattr(tensor, "original_hash", None) == source_hash), 0)
                ggml_prefetch_position = ggml_current_index % len(ggml_ring_buffer)
                ggml_prefetch_tensor = ggml_ring_buffer[ggml_prefetch_position]
                ggml_prefetch_tensor_hash = getattr(ggml_prefetch_tensor, "original_hash", None)

                if isinstance(ggml_prefetch_tensor_hash, torch.Tensor):
                    if ggml_prefetch_tensor_hash.numel() == 1:
                        ggml_prefetch_tensor_hash = ggml_prefetch_tensor_hash.item()
                    else:
                        ggml_prefetch_tensor_hash = tuple(ggml_prefetch_tensor_hash.tolist())

                ggml_prefetch_tensor = cached_tensor_map[ggml_prefetch_tensor_hash]['source_tensor'].to(tensorator_device, non_blocking=True)  # move tensor to tensorator_device while dequantizing and applying patches

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

                ggml_ring_buffer.append(ggml_prefetch_tensor)


            else:
                ggml_ring_buffer.pop(0)
                ggml_current_index = next((idx for idx, tensor in enumerate(ggml_ring_buffer) if getattr(tensor, "original_hash", None) == getattr(source_tensor, "original_hash", None)), 0) 
                ggml_prefetch_position = (ggml_current_index) % len(ggml_ring_buffer) # identify the position of the tensor in the tensor ring buffer
                ggml_prefetch_tensor_hash = ggml_ring_buffer[ggml_prefetch_position][0]
                ggml_prefetch_tensor = cached_tensor_map[ggml_prefetch_tensor_hash]['source_tensor'].to(tensorator_device, non_blocking=True)
                ggml_ring_buffer.append(ggml_prefetch_tensor)
                
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

            if source_tensor_hash in cached_tensor_map and cached_tensor_map[source_tensor_hash]['cache_level'] == "level1":                #second time through for a level1-assigned tensor as level 1 branches after the first time
                level_one_tensor = tensorator_tensor.clone().to(compute_device, non_blocking=True)
                level_one_tensors.append(level_one_tensor)
                cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = level_one_tensor
                #print(f"Moving Dequantized and Patched Tensor: 0x{source_tensor_hash:x} | tensor_inference_order: {cached_tensor_map[source_tensor_hash]['tensor_inference_order']:3d} | Size: {cached_tensor_map[source_tensor_hash]['tensor_size']:.2f} | to compute_device")
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
                return level_one_tensor
            elif source_tensor_hash in cached_tensor_map and cached_tensor_map[source_tensor_hash]['cache_level'] == "level2":
                level_two_tensor = tensorator_tensor.clone().to(tensorator_device, non_blocking=True)
                level_two_tensors.append(level_two_tensor)
                cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = level_two_tensor
                #print(f"Moving Dequantized and Patched Tensor: 0x{source_tensor_hash:x} | tensor_inference_order: {cached_tensor_map[source_tensor_hash]['tensor_inference_order']:3d} | Size: {cached_tensor_map[source_tensor_hash]['tensor_size']:.2f} | to tensorator_device")
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
                return level_two_tensor
            elif source_tensor_hash in cached_tensor_map and cached_tensor_map[source_tensor_hash]['cache_level'] == "tensor_ring":          #Assumes second video card is tensorator_device, should create a full-tensor ring buffer for compute_device 
                #print(f"Tensor in Tensor Ring Buffer: 0x{source_tensor_hash:x} | tensor_inference_order: {cached_tensor_map[source_tensor_hash]['tensor_inference_order']:3d} | Size: {cached_tensor_map[source_tensor_hash]['tensor_size']:.2f} | to tensorator_device")
                tensorator_event.record(tensorator_stream)
                torch.cuda.current_stream().wait_event(tensorator_event)
                return tensorator_tensor

            tensorator_tensor = tensorator_tensor.to(device=compute_device, non_blocking=True)
            tensorator_event.record(tensorator_stream)
        else:                                                                                                                                #First time through for a tensor
            if ggml_tensor is None:
                return

            patch_list = []
            for func, item, key in getattr(ggml_tensor, "patches", []):
                patches = retrieve_cached_patch(item, key)
                patch_list += patches
                    
            tensorator_tensor = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)

            if isinstance(tensorator_tensor, GGMLTensor):
                tensorator_tensor.__class__ = torch.Tensor

            if patch_list:
                if  patch_dtype is None:
                    tensorator_tensor = func(patch_list, tensorator_tensor, key)
                else:
                    patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                    tensorator_tensor = func(patch_list, tensorator_tensor, key, patch_dtype)
           
            if source_tensor_hash not in cached_tensor_map:
                cached_tensor_map[source_tensor_hash] = {}
                cached_tensor_map[source_tensor_hash]['tensor_inference_order'] = tensor_inference_order
                cached_tensor_map[source_tensor_hash]['patch_qty'] = len(patch_list)
                cached_tensor_map[source_tensor_hash]['tensor_size'] = (ggml_tensor.numel() * ggml_tensor.element_size() / (1024 * 1024))
                cached_tensor_map[source_tensor_hash]['distorch_device'] = distorch_device
                cached_tensor_map[source_tensor_hash]['cache_level'] = "uninitialized"
                cached_tensor_map[source_tensor_hash]['cached_final_tensor'] = None
                cached_tensor_map[source_tensor_hash]['name'] = name
                cached_tensor_map[source_tensor_hash]['dtype'] = dtype
                cached_tensor_map[source_tensor_hash]['dequant_dtype'] = dequant_dtype
                cached_tensor_map[source_tensor_hash]['patches'] = ggml_tensor.patches
                cached_tensor_map[source_tensor_hash]['patch_dtype'] = patch_dtype
                cached_tensor_map[source_tensor_hash]['source_tensor'] = source_tensor
                cached_tensor_map[source_tensor_hash]['final_tensor_size'] = tensorator_tensor.numel() * tensorator_tensor.element_size() / (1024 * 1024)
                #print(f"Caching Initialization: tensor_inference_order={tensor_inference_order} | patches={len(patch_list)} | device={distorch_device} | size={cached_tensor_map[source_tensor_hash]['tensor_size']:.2f}MB | name={name}")

  
    torch.cuda.current_stream().wait_event(tensorator_event)
    return tensorator_tensor