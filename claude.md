# ComfyUI-MultiGPU Development Guide

This guide documents the implementation of GGML_LOOK_AHEAD_CACHE for ComfyUI-MultiGPU.

## Working Philosophy

> "I work in steps not leaps. I know this is hard for you so that is why we are having this conversation. I will likely redirect you to do smaller and smaller chunks until I can understand what you are doing. The truth? I wrote none of this alone. I used llms for every line. That said, I wrote ALL of it, because I took the 1000s of lines of code and extracted only what was necessary in as elegant a manner as I am capable. Your code will not survie this. Our code will. I am a test engineer. I write optimal code = only what is necessary."

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

## GGML_LOOK_AHEAD_CACHE Implementation

The goal is to implement a simple non-blocking 30-tensor look-ahead caching system that sits in the compute device buffer. This will eliminate wait time for fetching tensors that are currently on CPU.

### Core Components

1. A fixed-size look-ahead buffer (30 tensors)
2. Non-blocking prefetch mechanism
3. Integration with existing GGUF loader
4. Proper memory management to avoid leaks

### Implementation Approach

1. Add a `register_patched_gguf_loader()` function that follows the same pattern as the existing `register_patched_ggufmodelpatcher()`
2. Modify the loader to implement prefetching of upcoming tensors
3. Store these tensors in a buffer on the compute device
4. Ensure non-blocking transfers to avoid impacting performance
5. Return cached tensors when available to eliminate wait time

### Expected Benefits

1. Eliminate CPU-to-GPU transfer latency for upcoming tensors
2. Improve overall inference throughput
3. Maintain compatibility with existing code
4. No impact on memory usage or other optimizations

This focused implementation provides immediate benefits while setting the stage for more advanced optimizations in the future.