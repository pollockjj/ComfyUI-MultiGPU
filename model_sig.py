def get_model_type(model_patcher):
    """
    Identifies the model type using a multi-layered approach for robustness.
    It first checks the diffusion model's class name, then falls back to the
    model_type enum.
    """
    if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
        class_name = type(model_patcher.model.diffusion_model).__name__
        
        # Prioritize class name for accuracy
        if "Flux" in class_name:
            return "FLUX"
        if "Qwen" in class_name:
            return "QWEN"
            
    # Fallback to the model_type enum for other cases
    if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'model_type'):
        # model_type is an enum, so we return its name as a string
        return model_patcher.model.model_type.name
        
    return "UNKNOWN"
