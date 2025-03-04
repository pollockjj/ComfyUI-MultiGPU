Total time: 205.586 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      13802.3      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      16538.8      0.5      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        113.5    113.5      0.0          print("\n" + "="*80)
   109         1         45.4     45.4      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         26.2     26.2      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         24.1     24.1      0.0          print("="*80 + "\n")
   112         1          0.6      0.6      0.0          get_weight._cuda1_logged = True
   113
   114     34048      53920.5      1.6      0.0      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5233.2      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4410.3      0.1      0.0      if use_cuda1_for_processing:
   123     34048      20759.2      0.6      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      48026.6      1.4      0.0          cuda1_device = torch.device("cuda:1")
   127     68096    1625316.6     23.9      0.8          with torch.cuda.stream(cuda1_stream):
   128     34048    2013357.9     59.1      1.0              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       6696.8      0.2      0.0              patch_list = []
   132     51072      20953.6      0.4      0.0              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134     17024    3935388.4    231.2      1.9                  patches = retrieve_cached_patch(item, cuda1_device, key)
   135     17024       7845.5      0.5      0.0                  patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3311605.5     97.3      1.6              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      13275.5      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       6096.7      0.2      0.0              if patch_list:
   144     17024       1841.1      0.1      0.0                  if patch_dtype is None:
   145     17024  192035478.0  11280.3     93.4                      w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148     17024       7599.3      0.4      0.0                  total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1678750.4     49.3      0.8              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      73496.9      2.2      0.0              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     618149.1     18.2      0.3          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      47408.0      1.4      0.0          nvtx.range_pop()  # end cuda1_processing
   160     34048      15673.9      0.5      0.0          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3906.6      0.1      0.0          return w
   164
   165                                               # Original implementation for when not using cuda:1 processing
   166                                               if not cache_config["use_tensor_cache"]:
   167                                                   nvtx.range_push("patch-transfer branch")
   168                                                   patch_list = []
   169                                                   d = tensor.device
   170                                                   for func, item, key in getattr(tensor, "patches", []):
   171                                                       patch_list += retrieve_cached_patch(item, d, key)
   172                                                   w = dequantize_tensor(tensor, dtype, dequant_dtype)
   173                                                   if GGMLTensor is not None and isinstance(w, GGMLTensor):
   174                                                       w.__class__ = torch.Tensor
   175                                                   if patch_list:
   176                                                       w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   177                                                       total_tensors_processed += 1
   178                                                   nvtx.range_pop()  # end patch-transfer branch
   179                                                   nvtx.range_pop()  # end get_weight entry
   180                                                   return w
   181
   182                                               # Full ping-pong caching branch (dynamically evaluated each call)
   183                                               if not hasattr(get_weight, "_first_call_logged"):
   184                                                   print("\n" + "="*80)
   185                                                   print("MultiGPU: Full tensor caching is ENABLED")
   186                                                   print(f"Using prefetch batch size of {prefetch_batch_size}")
   187                                                   print("="*80+"\n")
   188                                                   get_weight._first_call_logged = True
   189
   190                                               ptr = tensor.data_ptr()
   191                                               if ptr in cached_tensor_map and "level_zero_cache_location" in cached_tensor_map[ptr]:
   192                                                   nvtx.range_pop()
   193                                                   return cached_tensor_map[ptr]["level_zero_cache_location"]
   194
   195                                               buf = prefetch_buffers[active_buffer_index]
   196                                               if ptr in buf:
   197                                                   current_tensor_in_batch += 1
   198                                                   if current_tensor_in_batch == prefetch_batch_size // 2:
   199                                                       prefetch_next_batch()
   200                                                   if current_tensor_in_batch >= prefetch_batch_size:
   201                                                       torch.cuda.synchronize("cuda:0")
   202                                                       active_buffer_index = 1 - active_buffer_index
   203                                                       current_tensor_in_batch = 0
   204                                                   nvtx.range_pop()
   205                                                   return buf[ptr]
   206
   207                                               if ptr in cached_tensor_map and "level_two_cache_location" in cached_tensor_map[ptr]:
   208                                                   with torch.cuda.stream(cuda1_stream):
   209                                                       w = cached_tensor_map[ptr]["level_two_cache_location"]().clone()
   210                                                   ib = 1 - active_buffer_index
   211                                                   if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
   212                                                       next_batch_to_prefetch[ib].append(ptr)
   213                                                   nvtx.range_pop()
   214                                                   return w
   215
   216                                               patch_list = []
   217                                               d = tensor.device
   218                                               for func, item, key in getattr(tensor, "patches", []):
   219                                                   patch_list += retrieve_cached_patch(item, d, key)
   220                                               w = dequantize_tensor(tensor, dtype, dequant_dtype)
   221                                               if GGMLTensor is not None and isinstance(w, GGMLTensor):
   222                                                   w.__class__ = torch.Tensor
   223                                               if patch_list:
   224                                                   w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   225                                                   total_tensors_processed += 1
   226
   227                                               if ptr % 5 == 0 and use_level_zero_cache:
   228                                                   with torch.cuda.stream(cuda0_stream):
   229                                                       l0 = w.clone().to("cuda:0", non_blocking=True)
   230                                                   level_zero_tensors.append(l0)
   231                                                   if ptr not in cached_tensor_map:
   232                                                       cached_tensor_map[ptr] = {}
   233                                                   cached_tensor_map[ptr]["level_zero_cache_location"] = l0
   234                                               else:
   235                                                   with torch.cuda.stream(cuda1_stream):
   236                                                       l2 = w.clone().to("cuda:1", non_blocking=True)
   237                                                   cached_tensors.append(l2)
   238                                                   if ptr not in cached_tensor_map:
   239                                                       cached_tensor_map[ptr] = {}
   240                                                   cached_tensor_map[ptr]["level_two_cache_location"] = weakref.ref(l2)
   241                                                   ib = 1 - active_buffer_index
   242                                                   if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
   243                                                       next_batch_to_prefetch[ib].append(ptr)
   244
   245                                               nvtx.range_pop()  # end get_weight entry
   246                                               return w


  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:519 - weight_decompose
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:539 - pad_tensor_to_shape
  0.03 seconds - /home/johnj/ComfyUI/comfy/lora.py:331 - model_lora_keys_clip
  0.10 seconds - /home/johnj/ComfyUI/comfy/lora.py:409 - model_lora_keys_unet
  0.46 seconds - /home/johnj/ComfyUI/comfy/lora.py:144 - load_lora
191.32 seconds - /home/johnj/ComfyUI/comfy/lora.py:572 - calculate_weight
205.59 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight


Total time: 24.2557 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      12898.6      0.4      0.1      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      14894.4      0.4      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        131.2    131.2      0.0          print("\n" + "="*80)
   109         1         27.4     27.4      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         23.1     23.1      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         23.4     23.4      0.0          print("="*80 + "\n")
   112         1          0.3      0.3      0.0          get_weight._cuda1_logged = True
   113
   114     34048      55713.2      1.6      0.2      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5252.6      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4919.5      0.1      0.0      if use_cuda1_for_processing:
   123     34048      23290.3      0.7      0.1          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      48054.2      1.4      0.2          cuda1_device = torch.device("cuda:1")
   127     68096    1613028.3     23.7      6.7          with torch.cuda.stream(cuda1_stream):
   128     34048   11861070.2    348.4     48.9              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       6701.8      0.2      0.0              patch_list = []
   132     51072      19586.3      0.4      0.1              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134     17024    1899072.6    111.6      7.8                  patches = retrieve_cached_patch(item, cuda1_device, key)
   135     17024       6787.0      0.4      0.0                  patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3210077.4     94.3     13.2              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      14272.0      0.4      0.1              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       7180.2      0.2      0.0              if patch_list:
   144     17024       2128.5      0.1      0.0                  if patch_dtype is None:
   145     17024    3113905.1    182.9     12.8                      w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148     17024       7347.4      0.4      0.0                  total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1582327.1     46.5      6.5              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      67676.5      2.0      0.3              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     613986.2     18.0      2.5          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      45878.9      1.3      0.2          nvtx.range_pop()  # end cuda1_processing
   160     34048      15517.1      0.5      0.1          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3891.2      0.1      0.0          return w
   164
   165                                               # Original implementation for when not using cuda:1 processing
   166                                               if not cache_config["use_tensor_cache"]:
   167                                                   nvtx.range_push("patch-transfer branch")
   168                                                   patch_list = []
   169                                                   d = tensor.device
   170                                                   for func, item, key in getattr(tensor, "patches", []):
   171                                                       patch_list += retrieve_cached_patch(item, d, key)
   172                                                   w = dequantize_tensor(tensor, dtype, dequant_dtype)
   173                                                   if GGMLTensor is not None and isinstance(w, GGMLTensor):
   174                                                       w.__class__ = torch.Tensor
   175                                                   if patch_list:
   176                                                       w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   177                                                       total_tensors_processed += 1
   178                                                   nvtx.range_pop()  # end patch-transfer branch
   179                                                   nvtx.range_pop()  # end get_weight entry
   180                                                   return w
   181
   182                                               # Full ping-pong caching branch (dynamically evaluated each call)
   183                                               if not hasattr(get_weight, "_first_call_logged"):
   184                                                   print("\n" + "="*80)
   185                                                   print("MultiGPU: Full tensor caching is ENABLED")
   186                                                   print(f"Using prefetch batch size of {prefetch_batch_size}")
   187                                                   print("="*80+"\n")
   188                                                   get_weight._first_call_logged = True
   189
   190                                               ptr = tensor.data_ptr()
   191                                               if ptr in cached_tensor_map and "level_zero_cache_location" in cached_tensor_map[ptr]:
   192                                                   nvtx.range_pop()
   193                                                   return cached_tensor_map[ptr]["level_zero_cache_location"]
   194
   195                                               buf = prefetch_buffers[active_buffer_index]
   196                                               if ptr in buf:
   197                                                   current_tensor_in_batch += 1
   198                                                   if current_tensor_in_batch == prefetch_batch_size // 2:
   199                                                       prefetch_next_batch()
   200                                                   if current_tensor_in_batch >= prefetch_batch_size:
   201                                                       torch.cuda.synchronize("cuda:0")
   202                                                       active_buffer_index = 1 - active_buffer_index
   203                                                       current_tensor_in_batch = 0
   204                                                   nvtx.range_pop()
   205                                                   return buf[ptr]
   206
   207                                               if ptr in cached_tensor_map and "level_two_cache_location" in cached_tensor_map[ptr]:
   208                                                   with torch.cuda.stream(cuda1_stream):
   209                                                       w = cached_tensor_map[ptr]["level_two_cache_location"]().clone()
   210                                                   ib = 1 - active_buffer_index
   211                                                   if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
   212                                                       next_batch_to_prefetch[ib].append(ptr)
   213                                                   nvtx.range_pop()
   214                                                   return w
   215
   216                                               patch_list = []
   217                                               d = tensor.device
   218                                               for func, item, key in getattr(tensor, "patches", []):
   219                                                   patch_list += retrieve_cached_patch(item, d, key)
   220                                               w = dequantize_tensor(tensor, dtype, dequant_dtype)
   221                                               if GGMLTensor is not None and isinstance(w, GGMLTensor):
   222                                                   w.__class__ = torch.Tensor
   223                                               if patch_list:
   224                                                   w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   225                                                   total_tensors_processed += 1
   226
   227                                               if ptr % 5 == 0 and use_level_zero_cache:
   228                                                   with torch.cuda.stream(cuda0_stream):
   229                                                       l0 = w.clone().to("cuda:0", non_blocking=True)
   230                                                   level_zero_tensors.append(l0)
   231                                                   if ptr not in cached_tensor_map:
   232                                                       cached_tensor_map[ptr] = {}
   233                                                   cached_tensor_map[ptr]["level_zero_cache_location"] = l0
   234                                               else:
   235                                                   with torch.cuda.stream(cuda1_stream):
   236                                                       l2 = w.clone().to("cuda:1", non_blocking=True)
   237                                                   cached_tensors.append(l2)
   238                                                   if ptr not in cached_tensor_map:
   239                                                       cached_tensor_map[ptr] = {}
   240                                                   cached_tensor_map[ptr]["level_two_cache_location"] = weakref.ref(l2)
   241                                                   ib = 1 - active_buffer_index
   242                                                   if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
   243                                                       next_batch_to_prefetch[ib].append(ptr)
   244
   245                                               nvtx.range_pop()  # end get_weight entry
   246                                               return w


  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:519 - weight_decompose
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:539 - pad_tensor_to_shape
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:331 - model_lora_keys_clip
  0.01 seconds - /home/johnj/ComfyUI/comfy/lora.py:409 - model_lora_keys_unet
  0.06 seconds - /home/johnj/ComfyUI/comfy/lora.py:144 - load_lora
  2.94 seconds - /home/johnj/ComfyUI/comfy/lora.py:572 - calculate_weight
 24.26 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight


Total time: 18.6803 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      12748.4      0.4      0.1      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      15016.5      0.4      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        129.7    129.7      0.0          print("\n" + "="*80)
   109         1         29.3     29.3      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         45.3     45.3      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         28.8     28.8      0.0          print("="*80 + "\n")
   112         1          0.7      0.7      0.0          get_weight._cuda1_logged = True
   113
   114     34048      51974.0      1.5      0.3      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       4675.0      0.1      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4282.7      0.1      0.0      if use_cuda1_for_processing:
   123     34048      21073.3      0.6      0.1          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Step 1: Move GGML tensor to cuda:1 (using stream)
   126     34048      47203.5      1.4      0.3          cuda1_device = torch.device("cuda:1")
   127     68096    1604714.3     23.6      8.6          with torch.cuda.stream(cuda1_stream):
   128     34048   11386067.6    334.4     61.0              tensor_cuda1 = tensor.to(device=cuda1_device, non_blocking=True)
   129
   130                                                       # Step 2: Prepare patches on cuda:1
   131     34048       7281.1      0.2      0.0              patch_list = []
   132     34048      14445.8      0.4      0.1              for func, item, key in getattr(tensor, "patches", []):
   133                                                           # Use cuda:1 as target device for patches
   134                                                           patches = retrieve_cached_patch(item, cuda1_device, key)
   135                                                           patch_list += patches
   136
   137                                                       # Step 3: Dequantize on cuda:1
   138     34048    3224419.3     94.7     17.3              w = dequantize_tensor(tensor_cuda1, dtype, dequant_dtype)
   139     34048      14187.6      0.4      0.1              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   140                                                           w.__class__ = torch.Tensor
   141
   142                                                       # Step 4: Apply patches on cuda:1
   143     34048       7307.1      0.2      0.0              if patch_list:
   144                                                           if patch_dtype is None:
   145                                                               w = func(patch_list, w, key)
   146                                                           else:
   147                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   148                                                           total_tensors_processed += 1
   149
   150                                                       # Step 5: Transfer result back to cuda:0
   151     34048    1503528.6     44.2      8.0              w = w.to(device="cuda:0", non_blocking=True)
   152
   153                                                       # Record completion event
   154     34048      72896.7      2.1      0.4              cuda1_event.record(cuda1_stream)
   155
   156                                                   # Wait for cuda:1 to finish
   157     34048     630205.0     18.5      3.4          torch.cuda.current_stream().wait_event(cuda1_event)
   158
   159     34048      40625.0      1.2      0.2          nvtx.range_pop()  # end cuda1_processing
   160     34048      13524.1      0.4      0.1          nvtx.range_pop()  # end get_weight entry
   161
   162                                                   # Return the fully processed weight
   163     34048       3868.0      0.1      0.0          return w
   164
   165                                               # Original implementation for when not using cuda:1 processing
   166                                               if not cache_config["use_tensor_cache"]:
   167                                                   nvtx.range_push("patch-transfer branch")
   168                                                   patch_list = []
   169                                                   d = tensor.device
   170                                                   for func, item, key in getattr(tensor, "patches", []):
   171                                                       patch_list += retrieve_cached_patch(item, d, key)
   172                                                   w = dequantize_tensor(tensor, dtype, dequant_dtype)
   173                                                   if GGMLTensor is not None and isinstance(w, GGMLTensor):
   174                                                       w.__class__ = torch.Tensor
   175                                                   if patch_list:
   176                                                       w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   177                                                       total_tensors_processed += 1
   178                                                   nvtx.range_pop()  # end patch-transfer branch
   179                                                   nvtx.range_pop()  # end get_weight entry
   180                                                   return w
   181
   182                                               # Full ping-pong caching branch (dynamically evaluated each call)
   183                                               if not hasattr(get_weight, "_first_call_logged"):
   184                                                   print("\n" + "="*80)
   185                                                   print("MultiGPU: Full tensor caching is ENABLED")
   186                                                   print(f"Using prefetch batch size of {prefetch_batch_size}")
   187                                                   print("="*80+"\n")
   188                                                   get_weight._first_call_logged = True
   189
   190                                               ptr = tensor.data_ptr()
   191                                               if ptr in cached_tensor_map and "level_zero_cache_location" in cached_tensor_map[ptr]:
   192                                                   nvtx.range_pop()
   193                                                   return cached_tensor_map[ptr]["level_zero_cache_location"]
   194
   195                                               buf = prefetch_buffers[active_buffer_index]
   196                                               if ptr in buf:
   197                                                   current_tensor_in_batch += 1
   198                                                   if current_tensor_in_batch == prefetch_batch_size // 2:
   199                                                       prefetch_next_batch()
   200                                                   if current_tensor_in_batch >= prefetch_batch_size:
   201                                                       torch.cuda.synchronize("cuda:0")
   202                                                       active_buffer_index = 1 - active_buffer_index
   203                                                       current_tensor_in_batch = 0
   204                                                   nvtx.range_pop()
   205                                                   return buf[ptr]
   206
   207                                               if ptr in cached_tensor_map and "level_two_cache_location" in cached_tensor_map[ptr]:
   208                                                   with torch.cuda.stream(cuda1_stream):
   209                                                       w = cached_tensor_map[ptr]["level_two_cache_location"]().clone()
   210                                                   ib = 1 - active_buffer_index
   211                                                   if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
   212                                                       next_batch_to_prefetch[ib].append(ptr)
   213                                                   nvtx.range_pop()
   214                                                   return w
   215
   216                                               patch_list = []
   217                                               d = tensor.device
   218                                               for func, item, key in getattr(tensor, "patches", []):
   219                                                   patch_list += retrieve_cached_patch(item, d, key)
   220                                               w = dequantize_tensor(tensor, dtype, dequant_dtype)
   221                                               if GGMLTensor is not None and isinstance(w, GGMLTensor):
   222                                                   w.__class__ = torch.Tensor
   223                                               if patch_list:
   224                                                   w = func(patch_list, w, key) if patch_dtype is None else func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   225                                                   total_tensors_processed += 1
   226
   227                                               if ptr % 5 == 0 and use_level_zero_cache:
   228                                                   with torch.cuda.stream(cuda0_stream):
   229                                                       l0 = w.clone().to("cuda:0", non_blocking=True)
   230                                                   level_zero_tensors.append(l0)
   231                                                   if ptr not in cached_tensor_map:
   232                                                       cached_tensor_map[ptr] = {}
   233                                                   cached_tensor_map[ptr]["level_zero_cache_location"] = l0
   234                                               else:
   235                                                   with torch.cuda.stream(cuda1_stream):
   236                                                       l2 = w.clone().to("cuda:1", non_blocking=True)
   237                                                   cached_tensors.append(l2)
   238                                                   if ptr not in cached_tensor_map:
   239                                                       cached_tensor_map[ptr] = {}
   240                                                   cached_tensor_map[ptr]["level_two_cache_location"] = weakref.ref(l2)
   241                                                   ib = 1 - active_buffer_index
   242                                                   if len(next_batch_to_prefetch[ib]) < prefetch_batch_size:
   243                                                       next_batch_to_prefetch[ib].append(ptr)
   244
   245                                               nvtx.range_pop()  # end get_weight entry
   246                                               return w


  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:144 - load_lora
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:331 - model_lora_keys_clip
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:409 - model_lora_keys_unet
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:519 - weight_decompose
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:539 - pad_tensor_to_shape
  0.00 seconds - /home/johnj/ComfyUI/comfy/lora.py:572 - calculate_weight
 18.68 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
