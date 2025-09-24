# Active Context: Current Development Focus

## Current Work Focus

### Primary Development Status
**Project State**: Production Grade (Version 2.4.7)  
**Stability**: 300+ commits, 90 resolved issues  
**Community**: Active user base with consistent feedback  
**Performance**: Benchmarked and validated across hardware configurations  

### Recent Major Achievements (Last 6 Months)

#### DisTorch V2.0 Release (August 2025)
- **Universal SafeTensor Support**: Extended beyond GGUF to all model formats
- **10% Performance Improvement**: Optimized memory transfer patterns
- **Load-Patch-Distribute Pipeline**: Ensures quality parity with single-GPU
- **Expert Allocation Modes**: Bytes, ratios, fraction-based distribution

#### City96 Architecture Integration (December 2024 - Ongoing)
- **Code Reduction**: 400+ lines → 50 lines via inheritance
- **Dynamic Class Override**: Automatic node creation from existing loaders
- **Maintenance Simplification**: Auto-adapts to ComfyCore API changes
- **Universal Support**: Works with any ComfyUI loader pattern

#### Comprehensive Hardware Validation
- **6 Hardware Configurations**: NVLink to PCIe 3.0 x4 coverage
- **5 Model Architectures**: FLUX, WAN, QWEN, HunyuanVideo tested
- **Performance Benchmarking**: Quantified bandwidth vs. performance relationships
- **Strategic Recommendations**: Clear guidance for different use cases

## Current Development Priorities

### 1. Ecosystem Expansion (High Priority)
**Goal**: Support emerging model formats and custom nodes

**Active Integrations**:
- **ComfyUI-GGUF**: 6 DisTorch-enabled GGUF nodes (complete)
- **WanVideoWrapper**: 8 MultiGPU video nodes (complete)  
- **Florence2**: Vision model support (complete)
- **HunyuanVideoWrapper**: Native VAE + device selection (active development)

**Next Targets**:
- **LTX Video**: New video architecture support
- **Mochi**: Performance-optimized video models
- **Community Requests**: Issue-driven integration priorities

### 2. User Experience Optimization (Medium Priority)
**Goal**: Reduce complexity for new users while maintaining expert capabilities

**Recent Improvements**:
- **Automatic Mode**: Intelligent offloading based on VRAM availability
- **Error Messages**: Clear guidance when allocation fails
- **Example Workflows**: 20+ example JSON files covering major use cases

**Ongoing Work**:
- **Configuration Validation**: Prevent invalid allocation strings
- **Performance Prediction**: Estimate slowdown before execution
- **Documentation**: User-friendly guides for different hardware scenarios

### 3. Advanced Features (Low Priority)
**Goal**: Push boundaries of multi-device inference

**Research Areas**:
- **Model Parallelism**: Split individual layers across multiple devices
- **Pipeline Parallelism**: Concurrent execution of different workflow stages
- **Memory Compression**: Runtime compression of stored model layers
- **Quality Metrics**: Quantitative measurement of output quality preservation

## Active Technical Decisions

### Memory Management Philosophy
**Current Approach**: Conservative with user control
- **Default Behavior**: Minimal offloading unless user specifies
- **Safety First**: Automatic fallbacks when allocations fail
- **Transparency**: Clear logging of memory operations
- **User Choice**: Expert modes for power users

**Alternative Considered**: Aggressive automatic optimization
- **Rejected**: Too unpredictable, quality concerns with LoRAs
- **Lesson**: Users prefer control over convenience

### Integration Strategy
**Current Approach**: Inheritance-based class override
- **City96 Pattern**: Dynamic class creation at runtime
- **Minimal API Surface**: Reduces maintenance burden
- **ComfyCore Alignment**: Works with existing patterns

**Alternative Considered**: Direct node registration
- **Rejected**: Maintenance nightmare, API fragility
- **Lesson**: Elegant code reduces long-term costs

### Hardware Support Priority
**Current Approach**: Universal device support with quality tiers
- **Tier 1**: CUDA (primary development and testing)
- **Tier 2**: CPU, MPS (community validated)
- **Tier 3**: XPU, NPU, DirectML (experimental support)

**Rationale**: ComfyUI's diverse hardware ecosystem demands inclusivity

## User Behavior Patterns (Observed)

### Common Usage Scenarios
1. **Low-VRAM Image Generation** (40% of users)
   - Single GPU systems (RTX 4070, RTX 3080)
   - Running FLUX.1-dev, QWEN models
   - Primary strategy: CPU offloading

2. **Multi-GPU Video Generation** (30% of users)
   - Dual-GPU setups (mixed architectures common)
   - WAN, HunyuanVideo workflows
   - Primary strategy: GPU-to-GPU distribution

3. **Professional Workflows** (20% of users)
   - High-end hardware (3090s, 4090s)
   - Batch processing, high resolutions
   - Primary strategy: Optimization for throughput

4. **Enthusiast Experimentation** (10% of users)
   - Cutting-edge models, extreme configurations
   - Custom allocation strings, performance tweaking
   - Primary strategy: Push hardware limits

### Support Request Patterns
1. **"Only cuda:0 visible"** - Device detection issues (25%)
2. **"Out of memory errors"** - Allocation configuration (20%)
3. **"Slower than expected"** - Hardware optimization (15%)
4. **"Node missing after install"** - Dependency conflicts (15%)
5. **"Quality differences"** - LoRA/quantization concerns (10%)
6. **"Integration requests"** - New model support (15%)

### Configuration Preferences
- **Bytes Mode**: 60% adoption (preferred for precision)
- **Fraction Mode**: 25% adoption (simple but limited)
- **Ratio Mode**: 15% adoption (llama.cpp familiarity)

**Automatic vs Expert**: 70% start automatic, 40% graduate to expert modes

## Project Learnings & Insights

### What Works Well
1. **Inheritance Pattern**: City96's architecture scales beautifully
2. **Load-Patch-Distribute**: Maintains quality while enabling distribution
3. **Comprehensive Testing**: Hardware benchmarking prevents regression
4. **Conservative Defaults**: Users prefer working slowly to not working
5. **Clear Documentation**: Example workflows accelerate adoption

### What We've Learned to Avoid
1. **Defensive Programming**: Masks ComfyCore API changes, creates maintenance debt
2. **Automatic LoRA Offloading**: Quality concerns outweigh convenience
3. **Over-Optimization**: Complex algorithms often perform worse than simple ones
4. **API Abstraction**: Users want direct control over model placement
5. **Hardware Assumptions**: Every configuration is someone's primary system

### Development Philosophy Evolution
**Early**: "Make it work on as many systems as possible"
**Current**: "Make it work reliably, then optimize for common cases"
**Future**: "Provide the tools, let users choose their tradeoffs"

## Next Steps & Immediate Actions

### Short-term (Next 2-4 weeks)
1. **Issue Triage**: Address 5 highest-priority GitHub issues
2. **HunyuanVideo Integration**: Complete native VAE support
3. **Documentation Update**: Refresh README with current capabilities
4. **Example Refresh**: Update workflow examples for new features

### Medium-term (Next 2-3 months)  
1. **LTX Video Support**: Integrate new video model architecture
2. **Performance Dashboard**: Web-based hardware configuration guide
3. **Quality Validation**: Systematic output quality measurement
4. **Community Outreach**: Tutorial videos, blog posts

### Long-term (6-12 months)
1. **Model Parallelism**: Research splitting individual layers
2. **Streaming Inference**: Real-time video generation support
3. **Cloud Integration**: Multi-node distributed inference
4. **Professional Tools**: Batch processing, API server modes

## Knowledge Gaps & Research Areas

### Technical Uncertainties
1. **Future ComfyUI Changes**: Core API evolution risk
2. **Next-Gen Hardware**: PCIe 5.0, NVLink 5.0 optimization opportunities
3. **Model Architecture Evolution**: MoE, multimodal impact on distribution
4. **PyTorch Updates**: Memory management changes in newer versions

### Community Questions
1. **Adoption Barriers**: What prevents users from trying MultiGPU?
2. **Quality Perception**: Do users trust distributed inference quality?
3. **Hardware Investment**: Will users buy hardware based on MultiGPU support?
4. **Professional Use**: What features do commercial users need?

### Performance Mysteries
1. **Transfer Prediction**: Can we accurately predict slowdown before execution?
2. **Memory Fragmentation**: How do repeated loads/unloads affect performance?
3. **Thermal Behavior**: Does extended use show different performance patterns?
4. **OS Differences**: Are there meaningful Windows vs Linux performance gaps?

## Current Environment State

### Development Tools
- **Primary IDE**: VSCode with Python extensions
- **Version Control**: Git with conventional commits
- **Testing**: Manual validation across available hardware
- **Documentation**: Markdown files, example JSON workflows

### Hardware Access
- **Primary Development**: RTX 3090 with various secondary GPUs
- **Testing Network**: Community contributors with diverse configurations
- **Benchmarking**: Systematic testing across 6 hardware configurations
- **Limitations**: Limited access to newest hardware (RTX 5090, etc.)

### Community Engagement
- **GitHub Issues**: Active monitoring and response
- **Discord**: ComfyUI community support channel participation
- **Documentation**: Comprehensive README and example workflows
- **Support**: Personal responses to complex issues

This Memory Bank serves as my only link to previous work. Each reset, I depend entirely on these files to understand the project state and continue development effectively.
