# Design of Experiments (DOE) for Block Swap Optimization

## Overview
This document outlines the empirical optimization strategy for determining optimal block swap parameters based on actual hardware performance measurements rather than theoretical calculations.

## Objective
Find the optimal `swap_space_gb` parameter that minimizes total execution time for various model/hardware/workload combinations.

## Test Matrix Variables

### 1. Model Parameters
- **Active Parameter Size**: 1B, 7B, 13B, 24B, 70B parameters
- **Model Architecture**: 
  - SDXL (UNet-based)
  - Flux (Transformer-based)
  - HunyuanVideo (Transformer-based)
  - WanVideo (Transformer-based)

### 2. Latent Space Dimensions
#### Image Models
- 512x512 (SDXL base)
- 1024x1024 (SDXL highres)
- 2048x2048 (Flux highres)

#### Video Models
- 16 frames @ 512x512
- 49 frames @ 768x768
- 97 frames @ 1024x1024

### 3. Hardware Configurations
#### PCIe Generation
- PCIe 3.0 (16 GB/s)
- PCIe 4.0 (32 GB/s)
- PCIe 5.0 (64 GB/s)

#### GPU Memory Type
- GDDR6 (448 GB/s bandwidth)
- GDDR6X (672 GB/s bandwidth)
- HBM2 (900 GB/s bandwidth)
- HBM3 (3.2 TB/s bandwidth)

#### System Topology
- Single GPU
- Dual GPU (same PCIe root)
- Dual GPU (cross-socket)

### 4. Swap Configuration Test Points
```python
swap_space_test_points = [
    0.1,   # Minimal (DisTorch-like)
    0.25,  # 
    0.5,   # 
    1.0,   # Single block
    2.0,   # 
    4.0,   # 
    8.0,   # Large blocks
    16.0,  # Very large blocks
]
```

## Measurement Methodology

### Timing Measurements
```python
class BlockSwapBenchmark:
    def measure(self, model, swap_config):
        results = {
            "model_size_gb": self.get_model_size(model),
            "swap_space_gb": swap_config.swap_space,
            "virtual_vram_gb": swap_config.virtual_vram,
            
            # Timing breakdown
            "total_time": 0,
            "inference_time": 0,
            "transfer_time": 0,
            "overhead_time": 0,
            
            # Transfer statistics
            "num_transfers": 0,
            "avg_transfer_size_mb": 0,
            "peak_vram_usage_gb": 0,
            
            # Efficiency metrics
            "transfer_ratio": 0,  # transfer_time / total_time
            "compute_efficiency": 0,  # inference_time / total_time
        }
        
        # Run inference with instrumentation
        with self.timer() as t:
            # ... inference code ...
            pass
            
        return results
```

### Performance Metrics
1. **Primary Metric**: Total execution time (inference + transfers + overhead)
2. **Secondary Metrics**:
   - Transfer time ratio (% time spent in PCIe transfers)
   - Peak VRAM usage
   - Number of block swaps

## Empirical Results Table (To Be Populated)

| Model | Latent | PCIe | Swap Space | Total Time | Transfer % | Optimal |
|-------|--------|------|------------|------------|------------|---------|
| SDXL 2.1B | 1024x1024 | 4.0 | 0.1 GB | TBD | TBD | |
| SDXL 2.1B | 1024x1024 | 4.0 | 1.0 GB | TBD | TBD | ✓ |
| SDXL 2.1B | 1024x1024 | 4.0 | 4.0 GB | TBD | TBD | |
| Flux 12B | 1024x1024 | 4.0 | 0.1 GB | TBD | TBD | |
| Flux 12B | 1024x1024 | 4.0 | 2.0 GB | TBD | TBD | ✓ |
| HunyuanVideo 13B | 49 frames | 4.0 | 0.1 GB | TBD | TBD | ✓ |
| HunyuanVideo 13B | 49 frames | 4.0 | 4.0 GB | TBD | TBD | |

## Smart Defaults Implementation

### Regression Model
```python
def predict_optimal_swap_space(
    model_size_gb: float,
    latent_pixels: int,
    latent_frames: int,
    pcie_gen: float,
    gpu_bandwidth_gbps: float
) -> float:
    """
    Predict optimal swap space based on DOE results.
    
    Uses polynomial regression or lookup table interpolation
    based on empirical measurements.
    """
    
    # Video workloads: minimize transfers
    if latent_frames > 16:
        return 0.1  # Layer-by-layer (DisTorch mode)
    
    # Small models: can fit large blocks
    if model_size_gb < 2:
        return min(model_size_gb * 0.5, 4.0)
    
    # Interpolate from DOE results
    key = (model_size_gb, latent_pixels, pcie_gen)
    return interpolate_from_measurements(key, DOE_RESULTS)
```

### Auto Mode Configuration
```python
class AutoSwapConfig:
    def __init__(self):
        self.doe_results = self.load_doe_results()
        self.hardware_profile = self.detect_hardware()
    
    def get_optimal_config(self, model, workload):
        # Use empirical data to determine optimal settings
        model_size = self.get_model_size(model)
        latent_info = self.analyze_workload(workload)
        
        optimal_swap = self.predict_optimal_swap_space(
            model_size,
            latent_info,
            self.hardware_profile
        )
        
        return {
            "swap_space_gb": optimal_swap,
            "virtual_vram_gb": self.calculate_virtual_vram(model_size, optimal_swap),
            "confidence": self.get_prediction_confidence()
        }
```

## Benchmarking Harness

### Test Runner
```python
class DOETestRunner:
    def run_full_matrix(self):
        results = []
        
        for model in MODELS:
            for latent_config in LATENT_CONFIGS:
                for hardware in HARDWARE_CONFIGS:
                    for swap_space in SWAP_SPACE_POINTS:
                        result = self.run_single_test(
                            model, latent_config, hardware, swap_space
                        )
                        results.append(result)
                        
        self.save_results(results)
        self.generate_report(results)
```

### Community Contribution
Users can contribute their benchmark results:
```python
# Run benchmark on user's system
python -m comfyui_multigpu.benchmark --contribute

# Uploads anonymized results to help improve defaults
{
    "hardware_hash": "pcie4_rtx4090_64gbram",
    "model": "flux_schnell",
    "optimal_swap": 2.0,
    "speedup": 1.8
}
```

## Implementation Phases

### Phase 1: Basic Benchmarking (v2.1)
- Implement timing instrumentation
- Create simple test harness
- Gather initial measurements

### Phase 2: DOE Execution (v2.2)
- Run systematic tests
- Build results database
- Create regression model

### Phase 3: Auto Mode (v2.3)
- Implement prediction algorithm
- Add hardware detection
- Enable "Auto" option in nodes

### Phase 4: Community Optimization (v2.4)
- Add benchmark contribution system
- Continuous improvement of defaults
- Hardware-specific profiles

## Expected Outcomes

### For Image Generation (Short Inference)
- Larger swap spaces (2-4 GB) optimal
- Fewer, larger transfers
- 20-40% speedup expected

### For Video Generation (Long Inference)
- Minimal swap spaces (0.1-0.5 GB) optimal
- Transfer time < 1% of total
- Negligible performance difference

### Hardware Scaling
- PCIe 5.0: Can use larger blocks efficiently
- PCIe 3.0: Smaller blocks to minimize transfer impact
- HBM3: Can swap entire model quickly

## Notes

1. Initial implementation will use hardcoded defaults
2. DOE results will refine these over time
3. User override always available
4. Focus on 80/20 rule: optimize for common cases
