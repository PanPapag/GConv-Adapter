from typing import Any, Dict 


def serialize_finetuning_args(cfg: Dict[str, Any]) -> str:
    """
    Generate a run name based on the fine-tuning configuration provided.
    The function constructs run names based on the type of fine-tuning:
    - LoRA: Includes 'r', 'target_modules', and 'modules_to_save'.
    - BitFit: Includes 'target_modules'.
    - SurgicalFinetuning: Includes 'train_last_layer', 'train_layer_norm', 'train_batch_norm', and 'target_modules'.
    - CustomAdapter: Handles any adapter type, includes 'bottleneck_size', 'type', 'positions', and 'layers'.
    
    List values are joined by '&'. If 'target_modules' or 'modules_to_save' are empty or None, they default to 'all' and 'last_layer', respectively.
    
    Args:
        cfg (dict): Configuration dictionary containing the fine-tuning settings.
        
    Returns:
        str: A formatted run name.

    """
    
    target = cfg.get('_target_', '')

    if 'LoRA' in target:
        r = cfg.get('r', 'NA')
        target_modules = '&'.join(cfg.get('target_modules', ['all']) or ['all'])
        run_name = f"LoRA-r[{r}]-target_modules[{target_modules}]"
    
    elif 'BitFit' in target:
        target_modules = '&'.join(cfg.get('target_modules', ['all']) or ['all'])
        run_name = f"BitFit-target_modules[{target_modules}]"
    
    elif 'SurgicalFinetuning' in target:
        train_last_layer = 'True' if cfg.get('train_last_layer', False) else 'False'
        train_layer_norm = 'True' if cfg.get('train_layer_norm', False) else 'False'
        train_batch_norm = 'True' if cfg.get('train_batch_norm', False) else 'False'
        target_modules = '&'.join(cfg.get('target_modules', '')) or 'NA'
        run_name = f"SurgicalFinetuning-train_last_layer[{train_last_layer}]-train_layer_norm[{train_layer_norm}]-train_batch_norm[{train_batch_norm}]-target_modules[{target_modules}]"
    
    elif 'adapter' in cfg:
        adapter = cfg.get('adapter', {})
        adapter_type = adapter.get('_target_', '').split('.')[-1]
        bottleneck_size = adapter.get('bottleneck_size', 'NA')
        type_ = cfg.get('type', 'NA')
        positions = '&'.join(cfg.get('positions', ['NA']))
        run_name = f"{adapter_type}-bottleneck_size[{bottleneck_size}]-type[{type_}]-positions[{positions}]"
    
    else:
        run_name = "Unknown"
    
    return run_name


def serialize_dict_params(params: Dict[str, Any], mapping: Dict[str, str] = None) -> str:
    """
    Serialize dict parameters into a string and shorten specific keys.

    Args:
        params (dict): Dictionary of parameters.
        mapping (dict, optional): Dictionary to map specific keys to shorter strings.

    Returns:
        str: Serialized and shortened parameters string.
    """
    if mapping is None:
        mapping = {}

    serialized_parts = []
    for key, value in params.items():
        key = mapping.get(key, key)
        serialized_parts.append(f"{key}={value}")

    return "_".join(serialized_parts)