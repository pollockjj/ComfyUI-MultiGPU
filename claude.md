# ComfyUI-MultiGPU Development Guide

This guide captures the key architecture, decisions, and implementation details for ComfyUI-MultiGPU, specifically focusing on performance optimizations using dual GPUs.

## Working Philosophy

> "I work in steps not leaps. I know this is hard for you so that is why we are having this conversation. I will likely redirect you to do smaller and smaller chunks until I can understand what you are doing. The truth? I wrote none of this alone. I used llms for every line. That said, I wrote ALL of it, because I took the 1000s of lines of code and extracted only what was necessary in as elegant a manner as I am capable. Your code will not survie this. Our code will. I am a test engineer. I write optimal code = only what is necessary."

This philosophy guides our development approach - focused on iterative steps, understanding each component, extracting only what's necessary, and ensuring optimal efficiency in the final code.

### LLM Optimization Domain Limitations

A critical architectural challenge has emerged when using LLMs for high-performance computing and hardware optimization tasks. This codebase operates at the intersection of several specialized domains:

1. **Hardware-driven optimization**: Working directly with memory transfers, CUDA streams, and device management
2. **Block-oriented operations**: Operating on chunks of data rather than individual elements for bandwidth efficiency
3. **Deterministic memory management**: Precisely controlling when and where memory is allocated/released
4. **Performance-critical code paths**: Removing all unnecessary operations, checks, and conditional logic

These domains are significantly underrepresented in public codebases and thus in LLM training data. Most public repositories emphasize individual element operations, abundant error checking, and safety over raw performance - patterns that are actively harmful for this type of optimization work.

## ABSOLUTE CODE INTEGRITY REQUIREMENTS

1. NEVER add, remove, or modify ANY code without explicit instructions
2. PRESERVE EXACTLY the original code structure, including:
   - No additional comments
   - No variable renames
   - No whitespace changes
   - No added error handling
3. DO NOT EXTEND, ENHANCE OR COMPLETE functions beyond what is explicitly requested
4. ONLY MODIFY THE EXACT LINES specified in the instruction
5. NEVER CREATE NEW FUNCTIONS unless explicitly instructed
6. KEEP MODIFICATIONS MINIMAL - change only what is absolutely necessary
7. BEFORE BEGINNING ANY CODING:
   - Summarize the exact steps you plan to take
   - Get explicit approval from the user
   - Follow those approved steps precisely with no deviations

The above rules are NON-NEGOTIABLE. Violating these rules will result in IMMEDIATE PROJECT TERMINATION.

This codebase represents specialized hardware optimization techniques that may appear unusual but are carefully designed for maximum performance in GPU memory-limited environments. The architectural decisions must be honored precisely rather than "improved" with standard software patterns.


This the code we are currently working on.

        def new_gguf_sd_loader(path, handle_prefix="model.diffusion_model.", return_arch=False):

            reader = gguf.GGUFReader(path)

            # filter and strip prefix
            has_prefix = False
            if handle_prefix is not None:
                prefix_len = len(handle_prefix)
                tensor_names = set(tensor.name for tensor in reader.tensors)
                has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
                print(f"[PIPELINE MAP] handle_prefix={handle_prefix}, prefix_len={prefix_len}, has_prefix={has_prefix}, tensor_names_ptr=0x{id(tensor_names)&0xffff:04x}, count={len(tensor_names)}")


            tensors = []
            for tensor in reader.tensors:
                sd_key = tensor_name = tensor.name
                if has_prefix:
                    if not tensor_name.startswith(handle_prefix):
                        continue
                    sd_key = tensor_name[prefix_len:]
                tensors.append((sd_key, tensor))

            # detect and verify architecture
            compat = None
            arch_str = None
            arch_field = reader.get_field("general.architecture")
            if arch_field is not None:
                if len(arch_field.types) != 1 or arch_field.types[0] != gguf.GGUFValueType.STRING:
                    raise TypeError(f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}")
                arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
                if arch_str not in IMG_ARCH_LIST and arch_str not in TXT_ARCH_LIST:
                    raise ValueError(f"Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder but got {arch_str!r}")
            else: # stable-diffusion.cpp
                # Use pre-imported detect_arch function
                arch_str = detect_arch(set(val[0] for val in tensors)).arch
                compat = "sd.cpp"

            # main loading loop
            state_dict = {}
            qtype_dict = {}
            for sd_key, tensor in tensors:
                tensor_name = tensor.name
                tensor_type_str = str(tensor.tensor_type)
                torch_tensor = torch.from_numpy(tensor.data) # mmap

                shape = get_orig_shape(reader, tensor_name)
                if shape is None:
                    shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
                    # Workaround for stable-diffusion.cpp SDXL detection.
                    if compat == "sd.cpp" and arch_str == "sdxl":
                        if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                            while len(shape) > 2 and shape[-1] == 1:
                                shape = shape[:-1]

                # add to state dict
                if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                    torch_tensor = torch_tensor.view(*shape)
                state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
                qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

            # mark largest tensor for vram estimation
            qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
            if len(qsd) > 0:
                max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
                state_dict[max_key].is_largest_weight = True

            # sanity check debug print
            print("\nggml_sd_loader:")
            for k,v in qtype_dict.items():
                print(f" {k:30}{v:3}")

            if return_arch:
                return (state_dict, arch_str)
            return state_dict

        module.gguf_sd_loader = new_gguf_sd_loader
        module._loader_patched = True