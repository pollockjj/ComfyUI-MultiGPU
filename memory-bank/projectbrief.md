# ComfyUI-MultiGPU Project Brief

## Project Identity
**Name**: ComfyUI-MultiGPU  
**Maintainer**: John Pollock (@pollockjj)  
**Current Version**: 2.4.7 (Production Grade)  
**Repository**: https://github.com/pollockjj/ComfyUI-MultiGPU  

## Core Mission
Transform ComfyUI from single-GPU to multi-device AI inference platform. Stop using expensive compute cards for model storage - unleash them on maximum latent space instead.

## What We Build
A ComfyUI custom_node that provides:
- **Universal Multi-Device Support**: CUDA, CPU, XPU, NPU, MLU, MPS, DirectML
- **Advanced Memory Management**: DisTorch2 distributed model loading
- **Device-Aware Node Wrapping**: MultiGPU versions of all major ComfyUI loaders
- **Production-Grade Stability**: 300+ commits, 90 resolved issues

## Evolution Timeline
- **Aug 2024**: Basic multi-GPU device selection (Alexander Dzhoganov)
- **Dec 2024**: City96 architectural revolution (400+ lines → 50 lines via inheritance)
- **Jan 2025**: DisTorch V1 (GGUF virtual VRAM)
- **Aug 2025**: DisTorch V2.0 (Universal .safetensor support)
- **Sep 2025**: Production maturity (Version 2.4.7)

## Core Problems Solved
1. **VRAM Limitations**: Run 38GB models on 24GB cards
2. **Hardware Utilization**: Turn mixed GPU setups into unified compute pool
3. **Memory Management**: Deterministic model distribution vs dynamic --lowvram
4. **Workflow Scaling**: Enable previously impossible resolutions/batch sizes

## Primary User Segments
- **Low-VRAM Users**: 8GB-16GB cards accessing large models
- **Multi-GPU Enthusiasts**: 2x3090, mixed architecture setups
- **Production Users**: Consistent performance requirements
- **Video Generation**: WAN, HunyuanVideo, LTX workflows

## Technical Foundation
- **Dynamic Class Override System**: Elegant inheritance-based node wrapping
- **Load-Patch-Distribute (LPD)**: Load on compute → patch LoRAs → distribute at FP16
- **Virtual VRAM**: CPU/GPU memory appears as extended VRAM pool
- **Expert Allocation Modes**: Bytes, ratios, and fraction-based distribution

## Success Metrics
- **Community Adoption**: 300+ commits, active issue resolution
- **Performance Validation**: Benchmarked across hardware configurations
- **Ecosystem Integration**: Supports 15+ model loader types
- **Stability**: Production deployments running complex workflows

## Development Philosophy
- **Work WITH ComfyUI**: Leverage existing patterns, don't fight core
- **Fail Loudly**: No defensive coding - we want to know when ComfyCore changes
- **Self-Documenting Code**: Structure and names tell the story
- **Inheritance Over Composition**: Dynamic class overrides, not manual definitions
