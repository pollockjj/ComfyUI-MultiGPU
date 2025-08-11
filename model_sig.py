def get_model_type(model_patcher):
    """
    Identifies the model type by accessing the pre-determined model_type
    attribute from the model object.
    """
    if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'model_type'):
        # model_type is an enum, so we return its name as a string (e.g., "FLUX")
        return model_patcher.model.model_type.name
    return "UNKNOWN"
