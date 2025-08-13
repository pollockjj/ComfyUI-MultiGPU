"""
Device Memory Audit Utility for ComfyUI MultiGPU
Provides tools to inspect and monitor GPU/CPU memory usage and model placement
"""

import torch
import gc
import psutil
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import comfy.model_management as mm

def format_bytes(bytes_value: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def get_device_memory_info() -> Dict[str, Dict[str, Any]]:
    """
    Get current memory usage for all available devices
    
    Returns:
        Dict with device names as keys and memory info as values
    """
    memory_info = {}
    
    # Check CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = f"cuda:{i}"
            torch.cuda.synchronize(i)  # Ensure accurate memory reading
            
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            free = total - allocated
            
            memory_info[device_name] = {
                "allocated": allocated,
                "allocated_str": format_bytes(allocated),
                "reserved": reserved,
                "reserved_str": format_bytes(reserved),
                "total": total,
                "total_str": format_bytes(total),
                "free": free,
                "free_str": format_bytes(free),
                "usage_percent": (allocated / total * 100) if total > 0 else 0
            }
    
    # Check CPU/System memory
    vm = psutil.virtual_memory()
    memory_info["cpu"] = {
        "allocated": vm.used,
        "allocated_str": format_bytes(vm.used),
        "total": vm.total,
        "total_str": format_bytes(vm.total),
        "free": vm.available,
        "free_str": format_bytes(vm.available),
        "usage_percent": vm.percent
    }
    
    return memory_info

def audit_torch_tensors() -> Dict[str, List[Dict[str, Any]]]:
    """
    Find all torch tensors in memory and group by device
    
    Returns:
        Dict with device names as keys and list of tensor info as values
    """
    tensors_by_device = defaultdict(list)
    
    # Iterate through all objects in memory
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                device_str = str(obj.device)
                size_bytes = obj.element_size() * obj.numel()
                
                tensor_info = {
                    "id": id(obj),
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "size_bytes": size_bytes,
                    "size_str": format_bytes(size_bytes),
                    "requires_grad": obj.requires_grad,
                    "is_leaf": obj.is_leaf
                }
                
                tensors_by_device[device_str].append(tensor_info)
        except:
            # Some objects might not be accessible
            pass
    
    # Sort tensors by size for each device
    for device in tensors_by_device:
        tensors_by_device[device].sort(key=lambda x: x["size_bytes"], reverse=True)
    
    return dict(tensors_by_device)

def audit_model_placement(model) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    Audit where each layer of a model is placed
    
    Args:
        model: PyTorch model to audit
        
    Returns:
        Dict with device names as keys and list of (layer_name, layer_type, size) tuples
    """
    device_layers = defaultdict(list)
    
    for name, module in model.named_modules():
        # Skip container modules
        if len(list(module.children())) > 0:
            continue
            
        # Find device for this module
        device = None
        size_bytes = 0
        
        # Check parameters
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None:
                device = str(param.device)
                size_bytes += param.element_size() * param.numel()
        
        if device:
            layer_type = type(module).__name__
            device_layers[device].append((name, layer_type, size_bytes))
    
    return dict(device_layers)

def audit_comfy_models() -> Dict[str, Any]:
    """
    Audit ComfyUI's loaded models and their placement
    
    Returns:
        Dict with model information
    """
    model_info = {}
    
    # Try to get loaded models from ComfyUI's model management
    try:
        # Access currently loaded models if available
        if hasattr(mm, 'current_loaded_models'):
            loaded_models = mm.current_loaded_models
            
            for i, model_data in enumerate(loaded_models):
                model_name = f"model_{i}"
                if hasattr(model_data, 'model'):
                    model = model_data.model
                    placement = audit_model_placement(model)
                    
                    total_size = 0
                    for device_layers in placement.values():
                        for _, _, size in device_layers:
                            total_size += size
                    
                    model_info[model_name] = {
                        "type": type(model).__name__,
                        "device_placement": placement,
                        "total_size": total_size,
                        "total_size_str": format_bytes(total_size)
                    }
    except Exception as e:
        logging.warning(f"[MultiGPU] Could not audit ComfyUI models: {e}")
    
    return model_info

def print_memory_audit(detailed=False):
    """
    Print a formatted memory audit report
    
    Args:
        detailed: If True, include tensor-level details
    """
    logging.info("\n" + "=" * 60)
    logging.info("         DEVICE MEMORY AUDIT REPORT")
    logging.info("=" * 60)
    
    # Device memory overview
    memory_info = get_device_memory_info()
    logging.info("\n📊 MEMORY USAGE BY DEVICE:")
    logging.info("-" * 60)
    
    for device, info in memory_info.items():
        logging.info(f"\n{device.upper()}:")
        logging.info(f"  Total:     {info['total_str']:>12}")
        logging.info(f"  Allocated: {info['allocated_str']:>12} ({info['usage_percent']:.1f}%)")
        logging.info(f"  Free:      {info['free_str']:>12}")
        if 'reserved_str' in info:
            logging.info(f"  Reserved:  {info['reserved_str']:>12}")
    
    # Tensor audit
    if detailed:
        tensors = audit_torch_tensors()
        logging.info("\n📦 TENSORS BY DEVICE:")
        logging.info("-" * 60)
        
        for device, tensor_list in tensors.items():
            total_size = sum(t["size_bytes"] for t in tensor_list)
            logging.info(f"\n{device}: {len(tensor_list)} tensors, {format_bytes(total_size)} total")
            
            # Show top 5 largest tensors
            for i, tensor in enumerate(tensor_list[:5]):
                logging.info(f"  #{i+1}: Shape {tensor['shape']}, {tensor['size_str']}, {tensor['dtype']}")
    
    # ComfyUI model audit
    models = audit_comfy_models()
    if models:
        logging.info("\n🤖 COMFYUI LOADED MODELS:")
        logging.info("-" * 60)
        
        for model_name, info in models.items():
            logging.info(f"\n{model_name} ({info['type']}):")
            logging.info(f"  Total size: {info['total_size_str']}")
            
            for device, layers in info['device_placement'].items():
                device_size = sum(size for _, _, size in layers)
                logging.info(f"  {device}: {len(layers)} layers, {format_bytes(device_size)}")
    
    logging.info("\n" + "=" * 60)

def get_memory_snapshot() -> Dict[str, Any]:
    """
    Get a complete memory snapshot for benchmarking
    
    Returns:
        Dict containing all memory information
    """
    return {
        "device_memory": get_device_memory_info(),
        "tensors": audit_torch_tensors(),
        "models": audit_comfy_models()
    }

def compare_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two memory snapshots to see what changed
    
    Args:
        before: Snapshot taken before an operation
        after: Snapshot taken after an operation
        
    Returns:
        Dict with differences between snapshots
    """
    differences = {}
    
    # Compare device memory
    memory_diff = {}
    for device in after["device_memory"]:
        if device in before["device_memory"]:
            before_mem = before["device_memory"][device]["allocated"]
            after_mem = after["device_memory"][device]["allocated"]
            diff = after_mem - before_mem
            
            memory_diff[device] = {
                "before": format_bytes(before_mem),
                "after": format_bytes(after_mem),
                "difference": format_bytes(abs(diff)),
                "increased": diff > 0
            }
    
    differences["memory_changes"] = memory_diff
    
    # Compare tensor counts
    tensor_diff = {}
    for device in after["tensors"]:
        after_count = len(after["tensors"][device])
        before_count = len(before["tensors"].get(device, []))
        
        if after_count != before_count:
            tensor_diff[device] = {
                "before": before_count,
                "after": after_count,
                "difference": after_count - before_count
            }
    
    differences["tensor_count_changes"] = tensor_diff
    
    return differences

def print_snapshot_comparison(before: Dict[str, Any], after: Dict[str, Any], label: str = "Operation"):
    """
    Print a formatted comparison of two snapshots
    
    Args:
        before: Snapshot before operation
        after: Snapshot after operation
        label: Label for the operation being measured
    """
    diff = compare_snapshots(before, after)
    
    logging.info(f"\n📈 MEMORY CHANGES AFTER {label}:")
    logging.info("-" * 60)
    
    for device, changes in diff["memory_changes"].items():
        symbol = "↑" if changes["increased"] else "↓"
        logging.info(f"{device}: {changes['before']} → {changes['after']} ({symbol} {changes['difference']})")
    
    if diff["tensor_count_changes"]:
        logging.info("\nTensor count changes:")
        for device, changes in diff["tensor_count_changes"].items():
            logging.info(f"{device}: {changes['before']} → {changes['after']} ({changes['difference']:+d})")

# Example usage functions for benchmarking
def benchmark_memory_points():
    """
    Example function showing how to capture the 4 key memory points
    """
    logging.info("\n🔍 Starting Memory Benchmark...")
    
    # 1. Pre-load memory
    snapshot_pre = get_memory_snapshot()
    print_memory_audit(detailed=False)
    
    # User would load model here
    logging.info("\n[Load your model now]")
    input("Press Enter when model is loaded...")
    
    # 2. Post-load memory
    snapshot_post_load = get_memory_snapshot()
    print_snapshot_comparison(snapshot_pre, snapshot_post_load, "MODEL LOAD")
    
    # User would run inference here
    logging.info("\n[Run inference now]")
    input("Press Enter when inference is complete...")
    
    # 3. Active inference memory (captured during)
    snapshot_active = get_memory_snapshot()
    print_snapshot_comparison(snapshot_post_load, snapshot_active, "INFERENCE")
    
    # 4. Post-generation memory
    logging.info("\n[Waiting for cleanup...]")
    torch.cuda.empty_cache()  # Force cleanup
    gc.collect()
    
    snapshot_post_gen = get_memory_snapshot()
    print_snapshot_comparison(snapshot_active, snapshot_post_gen, "CLEANUP")
    
    logging.info("\n✅ Benchmark complete!")

if __name__ == "__main__":
    # Run a basic audit when script is executed directly
    print_memory_audit(detailed=True)
