Total time: 19.0902 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 90

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    90                                           @profile
    91                                           def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None, index=None, name=None, ggml_tensor_hash=None, distorch_device=None):
    92                                               global cached_tensor_map
    93     34048       6370.1      0.2      0.0      if ggml_tensor is None:
    94                                                   return None
    95
    96     34048       4943.6      0.1      0.0      patch_list = []
    97     34048     129328.5      3.8      0.7      device = ggml_tensor.device
    98     34048       8507.1      0.2      0.0      patches_data = getattr(ggml_tensor, "patches", [])
    99     51072      15639.5      0.3      0.1      for function, patches_item, key in patches_data:
   100     17024    1745671.4    102.5      9.1          patch_result = retrieve_cached_patch(patches_item, device, key)
   101     17024       8896.6      0.5      0.0          patch_list += patch_result
   102
   103
   104                                               # If cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized" then I want to set up the GGML buffer. This deterministic. I need the tensor on the CPU that is N positions ahead of the current index and we need to
   105                                               # prefetch it to a "ggml_prefetch_buffer" collection of hardrefs until we use it. So:
   106                                               # 1. Filter out all tensor that are not on CPU
   107                                               # 2. Sort the tensors by index
   108                                               # 3. Find the tensor that is N positions ahead of the current index
   109                                               # 4. Prefetch it to the "ggml_prefetch_buffer" collection of hardrefs, non-blocking
   110                                               # 5. Set the cache_level to "ggml_prefetch"
   111                                               # 6. Look to see if my own tensor is in the "ggml_prefetch_buffer" collection of hardrefs (which it will be after the first N tensors are prefetched)
   112                                               # 7. If it is swap out this reference for the one on remote, slow DRAM
   113                                               # 8. If it is not, do nothing. The normal flow will fetch the tensor (slowly, from DRAM) and process it as normal.
   114
   115     34048    3499112.7    102.8     18.3      weight = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)
   116
   117     34048       8318.3      0.2      0.0      if GGMLTensor is not None and isinstance(weight, GGMLTensor):
   118                                                   weight.__class__ = torch.Tensor
   119     34048       6740.3      0.2      0.0      if patch_list:
   120     17024       1741.4      0.1      0.0          if patch_dtype is None:
   121     17024   13614446.9    799.7     71.3              weight = function(patch_list, weight, key)
   122                                                   else:
   123                                                       computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
   124                                                       weight = function(patch_list, weight, key, computed_patch_dtype)
   125
   126     34048      14374.3      0.4      0.1      if ggml_tensor_hash not in cached_tensor_map and ggml_tensor_hash is not None:
   127       304        143.9      0.5      0.0          cached_tensor_map[ggml_tensor_hash] = {}
   128       304        109.3      0.4      0.0          cached_tensor_map[ggml_tensor_hash]['index'] = index
   129       304         68.6      0.2      0.0          cached_tensor_map[ggml_tensor_hash]['patch_qty'] = len(patch_list)
   130       304        137.8      0.5      0.0          cached_tensor_map[ggml_tensor_hash]['tensor_size'] = (ggml_tensor.numel() * ggml_tensor.element_size() / (1024 * 1024))
   131       304       2647.8      8.7      0.0          cached_tensor_map[ggml_tensor_hash]['distorch_device'] = distorch_device
   132       304         67.5      0.2      0.0          cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized"
   133       304        126.7      0.4      0.0          cached_tensor_map[ggml_tensor_hash]['cached_tensor'] = None
   134       304         59.0      0.2      0.0          cached_tensor_map[ggml_tensor_hash]['name'] = name
   135       304      18696.5     61.5      0.1          print(f"Recording order of Inference: ptr=0x{ggml_tensor_hash:x} | index={index} | patches={len(patch_list)} | device={distorch_device} | size={cached_tensor_map[ggml_tensor_hash]['tensor_size']:.2f}MB | name={name} ")
   136
   137     34048       4012.7      0.1      0.0      return weight


Total time: 258.992 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: cast_bias_weight_patched at line 44

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    44                                           @profile
    45                                           def cast_bias_weight_patched(s, input=None, dtype=None, device=None, bias_dtype=None):
    46                                               global cast_bias_weight_inf_ord
    47     17024      82167.7      4.8      0.0      from . import distorch_load_map
    48
    49     17024       3489.5      0.2      0.0      if input is not None:
    50     17024       2411.0      0.1      0.0          if dtype is None:
    51     17024      11609.6      0.7      0.0              dtype = getattr(input, "dtype", torch.float32)
    52     17024       1730.9      0.1      0.0          if bias_dtype is None:
    53     17024       1923.8      0.1      0.0              bias_dtype = dtype
    54     17024       1565.7      0.1      0.0          if device is None:
    55     17024       6299.2      0.4      0.0              device = input.device
    56
    57     17024      25099.4      1.5      0.0      ggml_tensor_hash = s.weight.original_hash
    58
    59     17024      11017.9      0.6      0.0      if ggml_tensor_hash in distorch_load_map and distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] is False:
    60       304        101.3      0.3      0.0          distorch_load_map[ggml_tensor_hash]['inf_order'] = cast_bias_weight_inf_ord
    61       304         99.5      0.3      0.0          cast_bias_weight_inf_ord += 1
    62       304         60.4      0.2      0.0          distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] = True
    63
    64
    65                                               # if cached_tensor_map[ggml_tensor_hash]['cache_level'] = "ggml_prefetch_buffer" we should skip.
    66     17024  238407257.8  14004.2     92.1      weight_to = s.weight.to(device)
    67
    68     17024       2713.5      0.2      0.0      bias = None
    69     17024      86761.0      5.1      0.0      non_blocking = comfy.model_management.device_supports_non_blocking(device)
    70     17024      32441.7      1.9      0.0      if s.bias is not None:
    71     17024    1483332.5     87.1      0.6          bias = s.get_weight(s.bias.to(device), dtype)
    72     17024     143611.0      8.4      0.1          bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)
    73
    74     17024       2366.5      0.1      0.0      kwargs = {}
    75     17024       8375.1      0.5      0.0      if ggml_tensor_hash in distorch_load_map and 'inf_order' in distorch_load_map[ggml_tensor_hash]:
    76     17024       6083.1      0.4      0.0          kwargs['index'] = distorch_load_map[ggml_tensor_hash]['inf_order']
    77     17024       2222.6      0.1      0.0      if ggml_tensor_hash in distorch_load_map:
    78     17024       5256.8      0.3      0.0          kwargs['name'] = distorch_load_map[ggml_tensor_hash]['name']
    79     17024       4059.3      0.2      0.0      if ggml_tensor_hash in distorch_load_map and 'distorch_device' in distorch_load_map[ggml_tensor_hash]:
    80     17024       4074.7      0.2      0.0          kwargs['distorch_device'] = distorch_load_map[ggml_tensor_hash]['distorch_device']
    81     17024       2895.8      0.2      0.0      kwargs['ggml_tensor_hash'] = ggml_tensor_hash
    82
    83     17024       1747.9      0.1      0.0      try:
    84     17024   18486840.7   1085.9      7.1          weight = s.get_weight(weight_to, dtype, **kwargs)
    85                                               except TypeError:
    86                                                   weight = s.get_weight(weight_to, dtype)
    87     17024     161606.5      9.5      0.1      weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    88
    89     17024       2774.9      0.2      0.0      return weight, bias
BASELINE


Total time: 19.3393 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 86

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    86                                           @profile
    87                                           def get_weight(ggml_tensor, dtype, dequant_dtype=None, patch_dtype=None, index=None, name=None, ggml_tensor_hash=None, distorch_device=None):
    88                                               global cached_tensor_map
    89     34048       6339.7      0.2      0.0      if ggml_tensor is None:
    90                                                   return None
    91
    92     34048       5451.1      0.2      0.0      patch_list = []
    93     34048     135552.5      4.0      0.7      device = ggml_tensor.device
    94     34048       9226.6      0.3      0.0      patches_data = getattr(ggml_tensor, "patches", [])
    95     51072      15608.5      0.3      0.1      for function, patches_item, key in patches_data:
    96     17024    1752273.8    102.9      9.1          patch_result = retrieve_cached_patch(patches_item, device, key)
    97     17024       8945.9      0.5      0.0          patch_list += patch_result
    98
    99
   100                                               # If cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized" then I want to set up the GGML buffer. This deterministic. I need the tensor on the CPU that is N positions ahead of the current index and we need to
   101                                               # prefetch it to a "ggml_prefetch_buffer" collection of hardrefs until we use it. So:
   102                                               # 1. Filter out all tensor that are not on CPU
   103                                               # 2. Sort the tensors by index
   104                                               # 3. Find the tensor that is N positions ahead of the current index
   105                                               # 4. Prefetch it to the "ggml_prefetch_buffer" collection of hardrefs, non-blocking
   106                                               # 5. Set the cache_level to "ggml_prefetch"
   107                                               # 6. Look to see if my own tensor is in the "ggml_prefetch_buffer" collection of hardrefs (which it will be after the first N tensors are prefetched)
   108                                               # 7. If it is swap out this reference for the one on remote, slow DRAM
   109                                               # 8. If it is not, do nothing. The normal flow will fetch the tensor (slowly, from DRAM) and process it as normal.
   110
   111     34048    3493721.6    102.6     18.1      weight = dequantize_tensor(ggml_tensor, dtype, dequant_dtype)
   112
   113     34048       8981.4      0.3      0.0      if GGMLTensor is not None and isinstance(weight, GGMLTensor):
   114                                                   weight.__class__ = torch.Tensor
   115     34048       6182.3      0.2      0.0      if patch_list:
   116     17024       2335.0      0.1      0.0          if patch_dtype is None:
   117     17024   13852904.3    813.7     71.6              weight = function(patch_list, weight, key)
   118                                                   else:
   119                                                       computed_patch_dtype = dtype if patch_dtype == "target" else patch_dtype
   120                                                       weight = function(patch_list, weight, key, computed_patch_dtype)
   121
   122     34048      15451.3      0.5      0.1      if ggml_tensor_hash not in cached_tensor_map and ggml_tensor_hash is not None:
   123       304        146.7      0.5      0.0          cached_tensor_map[ggml_tensor_hash] = {}
   124       304        111.4      0.4      0.0          cached_tensor_map[ggml_tensor_hash]['index'] = index
   125       304        124.3      0.4      0.0          cached_tensor_map[ggml_tensor_hash]['patch_qty'] = len(patch_list)
   126       304       2617.2      8.6      0.0          cached_tensor_map[ggml_tensor_hash]['tensor_size'] = (ggml_tensor.numel() * ggml_tensor.element_size() / (1024 * 1024))
   127       304         69.3      0.2      0.0          cached_tensor_map[ggml_tensor_hash]['distorch_device'] = distorch_device
   128       304         76.9      0.3      0.0          cached_tensor_map[ggml_tensor_hash]['cache_level'] = "uninitialized"
   129       304        109.6      0.4      0.0          cached_tensor_map[ggml_tensor_hash]['cached_tensor'] = None
   130       304         58.9      0.2      0.0          cached_tensor_map[ggml_tensor_hash]['name'] = name
   131       304      18983.5     62.4      0.1          print(f"Recording order of Inference: ptr=0x{ggml_tensor_hash:x} | index={index} | patches={len(patch_list)} | device={distorch_device} | size={cached_tensor_map[ggml_tensor_hash]['tensor_size']:.2f}MB | name={name} ")
   132
   133     34048       3983.9      0.1      0.0      return weight


Total time: 260.208 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: cast_bias_weight_patched at line 44

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    44                                           @profile
    45                                           def cast_bias_weight_patched(s, input=None, dtype=None, device=None, bias_dtype=None):
    46                                               global cast_bias_weight_inf_ord
    47     17024      82397.5      4.8      0.0      from . import distorch_load_map
    48
    49     17024       3773.2      0.2      0.0      if input is not None:
    50     17024       2206.6      0.1      0.0          if dtype is None:
    51     17024      12417.4      0.7      0.0              dtype = getattr(input, "dtype", torch.float32)
    52     17024       1611.6      0.1      0.0          if bias_dtype is None:
    53     17024       2077.8      0.1      0.0              bias_dtype = dtype
    54     17024       1483.2      0.1      0.0          if device is None:
    55     17024       6371.8      0.4      0.0              device = input.device
    56
    57     17024      24132.5      1.4      0.0      ggml_tensor_hash = s.weight.original_hash
    58
    59     17024      11588.1      0.7      0.0      if ggml_tensor_hash in distorch_load_map and distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] is False:
    60       304        112.8      0.4      0.0          distorch_load_map[ggml_tensor_hash]['inf_order'] = cast_bias_weight_inf_ord
    61       304         99.0      0.3      0.0          cast_bias_weight_inf_ord += 1
    62       304         64.4      0.2      0.0          distorch_load_map[ggml_tensor_hash]['cast_bias_weight'] = True
    63
    64     17024       2089.5      0.1      0.0      bias = None
    65     17024      83603.3      4.9      0.0      non_blocking = comfy.model_management.device_supports_non_blocking(device)
    66     17024      16216.2      1.0      0.0      if s.bias is not None:
    67     17024  169923099.7   9981.4     65.3          bias = s.get_weight(s.bias.to(device), dtype)
    68     17024     141253.0      8.3      0.1          bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)
    69
    70     17024       2088.5      0.1      0.0      kwargs = {}
    71     17024       7866.5      0.5      0.0      if ggml_tensor_hash in distorch_load_map and 'inf_order' in distorch_load_map[ggml_tensor_hash]:
    72     17024       5732.5      0.3      0.0          kwargs['index'] = distorch_load_map[ggml_tensor_hash]['inf_order']
    73     17024       2180.6      0.1      0.0      if ggml_tensor_hash in distorch_load_map:
    74     17024       5311.5      0.3      0.0          kwargs['name'] = distorch_load_map[ggml_tensor_hash]['name']
    75     17024       3224.6      0.2      0.0      if ggml_tensor_hash in distorch_load_map and 'distorch_device' in distorch_load_map[ggml_tensor_hash]:
    76     17024       3080.5      0.2      0.0          kwargs['distorch_device'] = distorch_load_map[ggml_tensor_hash]['distorch_device']
    77     17024       2557.4      0.2      0.0      kwargs['ggml_tensor_hash'] = ggml_tensor_hash
    78
    79     17024       1705.6      0.1      0.0      try:
    80     17024   89698316.9   5268.9     34.5          weight = s.get_weight(s.weight.to(device), dtype, **kwargs)
    81                                               except TypeError:
    82                                                   weight = s.get_weight(s.weight.to(device), dtype)
    83     17024     158668.0      9.3      0.1      weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
    84
    85     17024       2762.0      0.2      0.0      return weight, bias
Moved move

