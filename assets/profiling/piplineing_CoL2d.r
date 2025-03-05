GGML tensors on cuda0, no explict .to()

Total time: 159.385 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      12435.2      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      15036.8      0.4      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        126.9    126.9      0.0          print("\n" + "="*80)
   109         1         70.3     70.3      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         67.2     67.2      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         77.0     77.0      0.0          print("="*80 + "\n")
   112         1          0.9      0.9      0.0          get_weight._cuda1_logged = True
   113
   114     34048      50399.1      1.5      0.0      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4471.8      0.1      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4561.5      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20396.5      0.6      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      46826.5      1.4      0.0          cuda1_device = torch.device("cuda:1")
   127     68096    1586065.6     23.3      1.0          with torch.cuda.stream(cuda1_stream):
   128
   129                                                       # Step 2: Prepare patches on cuda:1
   130     34048       5593.4      0.2      0.0              patch_list = []
   131     51072      22433.7      0.4      0.0              for func, item, key in getattr(tensor, "patches", []):
   132                                                           # Use cuda:1 as target device for patches
   133     17024    3082375.2    181.1      1.9                  patches = retrieve_cached_patch(item, cuda1_device, key)
   134     17024       7683.4      0.5      0.0                  patch_list += patches
   135
   136                                                       # Step 3: Dequantize on cuda:1
   137     34048    7272903.2    213.6      4.6              w = dequantize_tensor(tensor, dtype, dequant_dtype)
   138     34048      14524.6      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   139                                                           w.__class__ = torch.Tensor
   140
   141                                                       # Step 4: Apply patches on cuda:1
   142     34048       6026.8      0.2      0.0              if patch_list:
   143     17024       1900.5      0.1      0.0                  if patch_dtype is None:
   144     17024  140735492.9   8266.9     88.3                      w = func(patch_list, w, key)
   145                                                           else:
   146                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   147     17024       8559.9      0.5      0.0                  total_tensors_processed += 1
   148
   149                                                       # Step 5: Transfer result back to cuda:0
   150     34048     740140.1     21.7      0.5              w = w.to(device="cuda:0", non_blocking=True)
   151
   152                                                       # Record completion event
   153     34048      89160.0      2.6      0.1              cuda1_event.record(cuda1_stream)
   154
   155                                                   # Wait for cuda:1 to finish
   156     34048    5590829.8    164.2      3.5          torch.cuda.current_stream().wait_event(cuda1_event)
   157
   158     34048      47818.7      1.4      0.0          nvtx.range_pop()  # end cuda1_processing
   159     34048      14841.2      0.4      0.0          nvtx.range_pop()  # end get_weight entry
   160
   161                                                   # Return the fully processed weight
   162     34048       4168.3      0.1      0.0          return w

159.38 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
200 seconds total 80%

Total time: 47.9617 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      13003.9      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      14503.3      0.4      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        128.4    128.4      0.0          print("\n" + "="*80)
   109         1         28.1     28.1      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         24.4     24.4      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         24.2     24.2      0.0          print("="*80 + "\n")
   112         1          0.5      0.5      0.0          get_weight._cuda1_logged = True
   113
   114     34048      50898.0      1.5      0.1      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4600.0      0.1      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4371.5      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20662.8      0.6      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      45065.7      1.3      0.1          cuda1_device = torch.device("cuda:1")
   127     68096    1561354.7     22.9      3.3          with torch.cuda.stream(cuda1_stream):
   128
   129                                                       # Step 2: Prepare patches on cuda:1
   130     34048       5280.5      0.2      0.0              patch_list = []
   131     51072      23284.1      0.5      0.0              for func, item, key in getattr(tensor, "patches", []):
   132                                                           # Use cuda:1 as target device for patches
   133     17024    1301905.8     76.5      2.7                  patches = retrieve_cached_patch(item, cuda1_device, key)
   134     17024       6656.7      0.4      0.0                  patch_list += patches
   135
   136                                                       # Step 3: Dequantize on cuda:1
   137     34048    8515666.7    250.1     17.8              w = dequantize_tensor(tensor, dtype, dequant_dtype)
   138     34048      14031.9      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   139                                                           w.__class__ = torch.Tensor
   140
   141                                                       # Step 4: Apply patches on cuda:1
   142     34048       6089.0      0.2      0.0              if patch_list:
   143     17024       1998.9      0.1      0.0                  if patch_dtype is None:
   144     17024   32720456.8   1922.0     68.2                      w = func(patch_list, w, key)
   145                                                           else:
   146                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   147     17024       7863.5      0.5      0.0                  total_tensors_processed += 1
   148
   149                                                       # Step 5: Transfer result back to cuda:0
   150     34048     718527.3     21.1      1.5              w = w.to(device="cuda:0", non_blocking=True)
   151
   152                                                       # Record completion event
   153     34048      84961.4      2.5      0.2              cuda1_event.record(cuda1_stream)
   154
   155                                                   # Wait for cuda:1 to finish
   156     34048    2779257.1     81.6      5.8          torch.cuda.current_stream().wait_event(cuda1_event)
   157
   158     34048      43462.4      1.3      0.1          nvtx.range_pop()  # end cuda1_processing
   159     34048      13666.0      0.4      0.0          nvtx.range_pop()  # end get_weight entry
   160
   161                                                   # Return the fully processed weight
   162     34048       3880.5      0.1      0.0          return w

 47.96 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
 88 seconds - 55%

 Total time: 9.62959 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      12506.7      0.4      0.1      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      13716.1      0.4      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        125.6    125.6      0.0          print("\n" + "="*80)
   109         1         28.0     28.0      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         24.7     24.7      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         25.3     25.3      0.0          print("="*80 + "\n")
   112         1          0.5      0.5      0.0          get_weight._cuda1_logged = True
   113
   114     34048      49903.8      1.5      0.5      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4942.9      0.1      0.1      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4387.1      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20229.6      0.6      0.2          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      44933.0      1.3      0.5          cuda1_device = torch.device("cuda:1")
   127     68096    1537716.9     22.6     16.0          with torch.cuda.stream(cuda1_stream):
   128
   129                                                       # Step 2: Prepare patches on cuda:1
   130     34048       5341.2      0.2      0.1              patch_list = []
   131     34048      16543.0      0.5      0.2              for func, item, key in getattr(tensor, "patches", []):
   132                                                           # Use cuda:1 as target device for patches
   133                                                           patches = retrieve_cached_patch(item, cuda1_device, key)
   134                                                           patch_list += patches
   135
   136                                                       # Step 3: Dequantize on cuda:1
   137     34048    6562403.6    192.7     68.1              w = dequantize_tensor(tensor, dtype, dequant_dtype)
   138     34048      13947.2      0.4      0.1              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   139                                                           w.__class__ = torch.Tensor
   140
   141                                                       # Step 4: Apply patches on cuda:1
   142     34048       6210.0      0.2      0.1              if patch_list:
   143                                                           if patch_dtype is None:
   144                                                               w = func(patch_list, w, key)
   145                                                           else:
   146                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   147                                                           total_tensors_processed += 1
   148
   149                                                       # Step 5: Transfer result back to cuda:0
   150     34048     627902.2     18.4      6.5              w = w.to(device="cuda:0", non_blocking=True)
   151
   152                                                       # Record completion event
   153     34048      82752.0      2.4      0.9              cuda1_event.record(cuda1_stream)
   154
   155                                                   # Wait for cuda:1 to finish
   156     34048     568606.5     16.7      5.9          torch.cuda.current_stream().wait_event(cuda1_event)
   157
   158     34048      40653.0      1.2      0.4          nvtx.range_pop()  # end cuda1_processing
   159     34048      12849.4      0.4      0.1          nvtx.range_pop()  # end get_weight entry
   160
   161                                                   # Return the fully processed weight
   162     34048       3841.7      0.1      0.0          return w

   9.63 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
70 seconds - 14%

