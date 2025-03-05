GGML tensors on cuda1, no explict .to()

8 LoRAs

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      14563.4      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      16868.4      0.5      0.0      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        126.9    126.9      0.0          print("\n" + "="*80)
   109         1         27.1     27.1      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         23.8     23.8      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         23.3     23.3      0.0          print("="*80 + "\n")
   112         1          0.5      0.5      0.0          get_weight._cuda1_logged = True
   113
   114     34048      59878.3      1.8      0.0      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5436.0      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       5907.8      0.2      0.0      if use_cuda1_for_processing:
   123     34048      20573.8      0.6      0.0          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Process directly on cuda:1
   126     34048      50504.7      1.5      0.0          cuda1_device = torch.device("cuda:1")
   127
   128     68096    1617185.4     23.7      0.5          with torch.cuda.stream(cuda1_stream):
   129                                                       # Step 2: Prepare patches on cuda:1
   130     34048       5651.7      0.2      0.0              patch_list = []
   131     51072      24139.6      0.5      0.0              for func, item, key in getattr(tensor, "patches", []):
   132                                                           # Use cuda:1 as target device for patches
   133     17024    5348888.0    314.2      1.8                  patches = retrieve_cached_patch(item, cuda1_device, key)
   134     17024       7940.2      0.5      0.0                  patch_list += patches
   135
   136                                                       # Step 3: Dequantize directly on cuda:1
   137     34048    3555304.8    104.4      1.2              w = dequantize_tensor(tensor, dtype, dequant_dtype)
   138     34048      14148.3      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   139                                                           w.__class__ = torch.Tensor
   140
   141                                                       # Step 4: Apply patches on cuda:1
   142     34048       6070.5      0.2      0.0              if patch_list:
   143     17024       2142.7      0.1      0.0                  if patch_dtype is None:
   144     17024  288926704.4  16971.7     95.9                      w = func(patch_list, w, key)
   145                                                           else:
   146                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   147     17024       8329.3      0.5      0.0                  total_tensors_processed += 1
   148
   149                                                       # Step 5: Transfer result back to cuda:0
   150     34048     720619.0     21.2      0.2              w = w.to(device="cuda:0", non_blocking=True)
   151
   152                                                       # Record completion event
   153     34048      77452.3      2.3      0.0              cuda1_event.record(cuda1_stream)
   154
   155                                                   # Wait for cuda:1 to finish
   156     34048     608940.5     17.9      0.2          torch.cuda.current_stream().wait_event(cuda1_event)
   157
   158     34048      48342.2      1.4      0.0          nvtx.range_pop()  # end cuda1_processing
   159     34048      15339.2      0.5      0.0          nvtx.range_pop()  # end get_weight entry
   160
   161                                                   # Return the fully processed weight
   162     34048       3925.1      0.1      0.0          return w



301.17 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
336 seconds, ~90% generation time% 


1 LoRA

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      14459.5      0.4      0.0      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      17759.4      0.5      0.1      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        140.7    140.7      0.0          print("\n" + "="*80)
   109         1         26.9     26.9      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         23.4     23.4      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         22.7     22.7      0.0          print("="*80 + "\n")
   112         1          0.7      0.7      0.0          get_weight._cuda1_logged = True
   113
   114     34048      61570.9      1.8      0.2      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5729.6      0.2      0.0      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4860.2      0.1      0.0      if use_cuda1_for_processing:
   123     34048      22284.8      0.7      0.1          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Process directly on cuda:1
   126     34048      51670.3      1.5      0.2          cuda1_device = torch.device("cuda:1")
   127
   128     68096    1636615.2     24.0      4.9          with torch.cuda.stream(cuda1_stream):
   129                                                       # Step 2: Prepare patches on cuda:1
   130     34048       6263.5      0.2      0.0              patch_list = []
   131     51072      23644.1      0.5      0.1              for func, item, key in getattr(tensor, "patches", []):
   132                                                           # Use cuda:1 as target device for patches
   133     17024    1663490.9     97.7      5.0                  patches = retrieve_cached_patch(item, cuda1_device, key)
   134     17024       6950.0      0.4      0.0                  patch_list += patches
   135
   136                                                       # Step 3: Dequantize directly on cuda:1
   137     34048   15075885.2    442.8     44.9              w = dequantize_tensor(tensor, dtype, dequant_dtype)
   138     34048      13781.5      0.4      0.0              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   139                                                           w.__class__ = torch.Tensor
   140
   141                                                       # Step 4: Apply patches on cuda:1
   142     34048       5892.0      0.2      0.0              if patch_list:
   143     17024       1894.6      0.1      0.0                  if patch_dtype is None:
   144     17024    7644419.6    449.0     22.8                      w = func(patch_list, w, key)
   145                                                           else:
   146                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   147     17024       8365.6      0.5      0.0                  total_tensors_processed += 1
   148
   149                                                       # Step 5: Transfer result back to cuda:0
   150     34048     739441.0     21.7      2.2              w = w.to(device="cuda:0", non_blocking=True)
   151
   152                                                       # Record completion event
   153     34048      81771.4      2.4      0.2              cuda1_event.record(cuda1_stream)
   154
   155                                                   # Wait for cuda:1 to finish
   156     34048    6431312.4    188.9     19.2          torch.cuda.current_stream().wait_event(cuda1_event)
   157
   158     34048      45150.2      1.3      0.1          nvtx.range_pop()  # end cuda1_processing
   159     34048      14491.9      0.4      0.0          nvtx.range_pop()  # end get_weight entry
   160
   161                                                   # Return the fully processed weight
   162     34048       3893.4      0.1      0.0          return w


 33.58 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
112 total time, ~30% generation time

0 LoRAs

Total time: 8.39847 s
File: /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py
Function: get_weight at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                           @profile
   100                                           def get_weight(tensor, dtype, dequant_dtype=None, patch_dtype=None):
   101                                               global total_tensors_processed, active_buffer_index, current_tensor_in_batch, next_batch_to_prefetch
   102
   103                                               # Use configuration option for cuda:1 processing
   104     34048      14519.6      0.4      0.2      use_cuda1_for_processing = cache_config['use_cuda1_processing']
   105
   106                                               # Print message when cuda:1 processing is first used
   107     34048      16622.7      0.5      0.2      if use_cuda1_for_processing and not hasattr(get_weight, "_cuda1_logged"):
   108         1        126.1    126.1      0.0          print("\n" + "="*80)
   109         1         27.5     27.5      0.0          print("MultiGPU: CUDA:1 Processing ENABLED")
   110         1         24.8     24.8      0.0          print("Dequantizing and patching on CUDA:1 before transferring to CUDA:0")
   111         1         24.2     24.2      0.0          print("="*80 + "\n")
   112         1          0.4      0.4      0.0          get_weight._cuda1_logged = True
   113
   114     34048      61030.0      1.8      0.7      nvtx.range_push("get_weight entry")
   115
   116                                               # Check if tensor is None
   117     34048       5705.0      0.2      0.1      if tensor is None:
   118                                                   nvtx.range_pop()  # end get_weight entry
   119                                                   return None
   120
   121                                               # Phase 1: Basic Linear Pipeline Implementation
   122     34048       4680.0      0.1      0.1      if use_cuda1_for_processing:
   123     34048      22612.0      0.7      0.3          nvtx.range_push("cuda1_processing")
   124
   125                                                   # Process directly on cuda:1
   126     34048      49616.0      1.5      0.6          cuda1_device = torch.device("cuda:1")
   127
   128     68096    1622278.3     23.8     19.3          with torch.cuda.stream(cuda1_stream):
   129                                                       # Step 2: Prepare patches on cuda:1
   130     34048       5485.4      0.2      0.1              patch_list = []
   131     34048      17282.3      0.5      0.2              for func, item, key in getattr(tensor, "patches", []):
   132                                                           # Use cuda:1 as target device for patches
   133                                                           patches = retrieve_cached_patch(item, cuda1_device, key)
   134                                                           patch_list += patches
   135
   136                                                       # Step 3: Dequantize directly on cuda:1
   137     34048    5189657.9    152.4     61.8              w = dequantize_tensor(tensor, dtype, dequant_dtype)
   138     34048      13360.8      0.4      0.2              if GGMLTensor is not None and isinstance(w, GGMLTensor):
   139                                                           w.__class__ = torch.Tensor
   140
   141                                                       # Step 4: Apply patches on cuda:1
   142     34048       7010.6      0.2      0.1              if patch_list:
   143                                                           if patch_dtype is None:
   144                                                               w = func(patch_list, w, key)
   145                                                           else:
   146                                                               w = func(patch_list, w, key, dtype if patch_dtype=="target" else patch_dtype)
   147                                                           total_tensors_processed += 1
   148
   149                                                       # Step 5: Transfer result back to cuda:0
   150     34048     644395.5     18.9      7.7              w = w.to(device="cuda:0", non_blocking=True)
   151
   152                                                       # Record completion event
   153     34048      80807.8      2.4      1.0              cuda1_event.record(cuda1_stream)
   154
   155                                                   # Wait for cuda:1 to finish
   156     34048     583417.2     17.1      6.9          torch.cuda.current_stream().wait_event(cuda1_event)
   157
   158     34048      41869.5      1.2      0.5          nvtx.range_pop()  # end cuda1_processing
   159     34048      13986.3      0.4      0.2          nvtx.range_pop()  # end get_weight entry
   160
   161                                                   # Return the fully processed weight
   162     34048       3929.0      0.1      0.0          return w

  8.40 seconds - /home/johnj/ComfyUI/custom_nodes/ComfyUI-MultiGPU/ggml_weight_utils.py:99 - get_weight
82 seconds. ~10% generation time