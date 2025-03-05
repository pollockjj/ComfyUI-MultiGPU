GGML tensors on cuda0, explict tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)

8 LoRAs

TTotal time: 212.49 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      12492.9      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      15015.4      0.4      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        127.8    127.8      0.0          print("\n" + "="*80)
   109         1         75.7     75.7      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         26.7     26.7      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         26.1     26.1      0.0          print("="*80 + "\n")
   112         1          0.4      0.4      0.0          get_weight._cuda1_logged = True
   113
   114     34048      54891.3      1.6      0.0      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5136.6      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4969.4      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20926.8      0.6      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      48168.5      1.4      0.0          cuda1_device = torch.device("cuda:1")
   127     68096    1616568.3     23.7      0.8          with torch.cuda.stream(cuda1_stream):
   128     34048    1880947.2     55.2      0.9              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       5902.0      0.2      0.0              patch_list = []
   132     51072      21038.6      0.4      0.0              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134     17024    4580311.1    269.1      2.2                  patches = retrieve_cached_patch(item, cuda1_device, key)
   135     17024       7961.8      0.5      0.0                  patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    4559862.7    133.9      2.1              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      13399.9      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       6635.2      0.2      0.0              if patch_list:
   144     17024       2127.8      0.1      0.0                  if patch_dtype is None:
   145     17024  187922298.4  11038.7     88.4                      w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148     17024       8730.5      0.5      0.0                  total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048   10917206.8    320.6      5.1              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      76613.7      2.3      0.0              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     647366.6     19.0      0.3          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      44088.3      1.3      0.0          nvtx.range_pop()  # end cuda1_processing
   160     34048      13500.2      0.4      0.0          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3836.6      0.1      0.0          return w

212.49 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
236 90%

Total time: 43.7008 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      13793.0      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      15005.8      0.4      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        129.3    129.3      0.0          print("\n" + "="*80)
   109         1         45.0     45.0      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         39.0     39.0      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         41.7     41.7      0.0          print("="*80 + "\n")
   112         1          0.6      0.6      0.0          get_weight._cuda1_logged = True
   113
   114     34048      52625.6      1.5      0.1      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4646.1      0.1      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4741.9      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20813.8      0.6      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      48309.5      1.4      0.1          cuda1_device = torch.device("cuda:1")
   127     68096    1598255.2     23.5      3.7          with torch.cuda.stream(cuda1_stream):
   128     34048    9960928.6    292.6     22.8              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       5705.6      0.2      0.0              patch_list = []
   132     51072      20430.1      0.4      0.0              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134     17024    1924742.2    113.1      4.4                  patches = retrieve_cached_patch(item, cuda1_device, key)
   135     17024       6785.8      0.4      0.0                  patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    4786028.2    140.6     11.0              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      14605.9      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       6156.0      0.2      0.0              if patch_list:
   144     17024       1721.3      0.1      0.0                  if patch_dtype is None:
   145     17024   22006946.3   1292.7     50.4                      w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148     17024       7989.4      0.5      0.0                  total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    2439021.8     71.6      5.6              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      69666.2      2.0      0.2              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     631000.8     18.5      1.4          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      43285.5      1.3      0.1          nvtx.range_pop()  # end cuda1_processing
   160     34048      13391.0      0.4      0.0          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3919.8      0.1      0.0          return w
   
 43.70 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
124 seconds, 45%

Total time: 18.4428 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      12495.2      0.4      0.1      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      15634.1      0.5      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        130.6    130.6      0.0          print("\n" + "="*80)
   109         1         45.8     45.8      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         27.7     27.7      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         27.8     27.8      0.0          print("="*80 + "\n")
   112         1          0.5      0.5      0.0          get_weight._cuda1_logged = True
   113
   114     34048      52275.7      1.5      0.3      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4452.3      0.1      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4581.8      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20064.3      0.6      0.1          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      46940.7      1.4      0.3          cuda1_device = torch.device("cuda:1")
   127     68096    1586419.1     23.3      8.6          with torch.cuda.stream(cuda1_stream):
   128     34048   11209081.6    329.2     60.8              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       5877.4      0.2      0.0              patch_list = []
   132     34048      13875.8      0.4      0.1              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134                                                           patches = retrieve_cached_patch(item, cuda1_device, key)
   135                                                           patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3192977.5     93.8     17.3              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      12806.5      0.4      0.1              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       5840.7      0.2      0.0              if patch_list:
   144                                                           if patch_dtype is None:
   145                                                               w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148                                                           total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1507142.3     44.3      8.2              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      69665.6      2.0      0.4              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     622933.3     18.3      3.4          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      42198.8      1.2      0.2          nvtx.range_pop()  # end cuda1_processing
   160     34048      13392.0      0.4      0.1          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3876.0      0.1      0.0          return w


 18.44 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
(.venv) johnj@llamatron:~/ComfyUI$
107 seconds - 17%%