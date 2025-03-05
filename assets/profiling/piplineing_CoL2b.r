GGML tensors on cuda1, explict tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)

8 LoRAs

Total time: 217.884 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      14664.3      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      18209.0      0.5      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        124.3    124.3      0.0          print("\n" + "="*80)
   109         1         26.9     26.9      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         42.7     42.7      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         52.6     52.6      0.0          print("="*80 + "\n")
   112         1          1.3      1.3      0.0          get_weight._cuda1_logged = True
   113
   114     34048      64110.3      1.9      0.0      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5317.8      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       5166.6      0.2      0.0      if use_cuda1_for_processing:
   123     34048      22423.2      0.7      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      50190.2      1.5      0.0          cuda1_device = torch.device("cuda:1")
   127     68096    1640725.4     24.1      0.8          with torch.cuda.stream(cuda1_stream):
   128     34048    1656120.6     48.6      0.8              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       6500.6      0.2      0.0              patch_list = []
   132     51072      21046.6      0.4      0.0              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134     17024    4194234.4    246.4      1.9                  patches = retrieve_cached_patch(item, cuda1_device, key)
   135     17024       7892.7      0.5      0.0                  patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3287652.8     96.6      1.5              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      13828.2      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       6643.1      0.2      0.0              if patch_list:
   144     17024       1939.2      0.1      0.0                  if patch_dtype is None:
   145     17024  204516194.0  12013.4     93.9                      w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148     17024       8041.6      0.5      0.0                  total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1602425.3     47.1      0.7              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      65304.2      1.9      0.0              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     611478.7     18.0      0.3          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      45406.7      1.3      0.0          nvtx.range_pop()  # end cuda1_processing
   160     34048      14442.4      0.4      0.0          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3963.3      0.1      0.0          return w


217.88 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
249 total generation time, 87%

1 LoRA
Total time: 17.1343 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      14124.2      0.4      0.1      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      18155.0      0.5      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        125.0    125.0      0.0          print("\n" + "="*80)
   109         1         27.1     27.1      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         23.0     23.0      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         23.0     23.0      0.0          print("="*80 + "\n")
   112         1          0.3      0.3      0.0          get_weight._cuda1_logged = True
   113
   114     34048      60170.5      1.8      0.4      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4971.6      0.1      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       5108.8      0.2      0.0      if use_cuda1_for_processing:
   123     34048      22118.1      0.6      0.1          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      51496.7      1.5      0.3          cuda1_device = torch.device("cuda:1")
   127     68096    1623573.2     23.8      9.5          with torch.cuda.stream(cuda1_stream):
   128     34048    4595202.2    135.0     26.8              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       6157.4      0.2      0.0              patch_list = []
   132     51072      21005.1      0.4      0.1              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134     17024    2119751.5    124.5     12.4                  patches = retrieve_cached_patch(item, cuda1_device, key)
   135     17024       6871.4      0.4      0.0                  patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3262260.9     95.8     19.0              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      14744.8      0.4      0.1              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       6560.7      0.2      0.0              if patch_list:
   144     17024       1960.6      0.1      0.0                  if patch_dtype is None:
   145     17024    2955019.2    173.6     17.2                      w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148     17024       8258.8      0.5      0.0                  total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1596691.7     46.9      9.3              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      66996.3      2.0      0.4              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     607024.5     17.8      3.5          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      47737.3      1.4      0.3          nvtx.range_pop()  # end cuda1_processing
   160     34048      14242.6      0.4      0.1          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3861.7      0.1      0.0          return w
   

 17.13 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
138 total seconds, =12%

0 Loras

Total time: 11.5541 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      14736.8      0.4      0.1      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      16818.2      0.5      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        125.3    125.3      0.0          print("\n" + "="*80)
   109         1         26.1     26.1      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         22.4     22.4      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         26.1     26.1      0.0          print("="*80 + "\n")
   112         1          0.3      0.3      0.0          get_weight._cuda1_logged = True
   113
   114     34048      61244.5      1.8      0.5      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5145.8      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       5531.1      0.2      0.0      if use_cuda1_for_processing:
   123     34048      21001.4      0.6      0.2          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      52106.0      1.5      0.5          cuda1_device = torch.device("cuda:1")
   127     68096    1646000.6     24.2     14.2          with torch.cuda.stream(cuda1_stream):
   128     34048    4164444.4    122.3     36.0              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       5751.1      0.2      0.0              patch_list = []
   132     34048      13371.0      0.4      0.1              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134                                                           patches = retrieve_cached_patch(item, cuda1_device, key)
   135                                                           patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3255628.2     95.6     28.2              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      13285.0      0.4      0.1              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       6199.1      0.2      0.1              if patch_list:
   144                                                           if patch_dtype is None:
   145                                                               w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148                                                           total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1516118.6     44.5     13.1              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      65730.6      1.9      0.6              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     632034.5     18.6      5.5          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      42094.6      1.2      0.4          nvtx.range_pop()  # end cuda1_processing
   160     34048      12850.6      0.4      0.1          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3764.8      0.1      0.0          return w
   


 11.55 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
120 seconds total time 9.625%%

