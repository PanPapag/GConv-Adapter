import logging
import torch
import torch.nn as nn
import pandas as pd


def summarize_model(model: nn.Module) -> pd.DataFrame:
    """
    Summarize the parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to summarize.

    Returns:
        pd.DataFrame: A DataFrame containing a summary of the model's parameters,
                      including the module names, shapes, number of parameters, 
                      and whether they are trainable or not. Additionally, includes
                      a row summarizing the total number of parameters and trainable parameters.
    """
    summary = []
    
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for name, param in model.named_parameters():
        param_details = {}
        param_details["Module"] = name
        param_details["Shape"] = list(param.size())
        param_details["Num Parameters"] = param.numel()
        param_details["Trainable"] = 'Yes' if param.requires_grad else 'No'
        
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()
        
        summary.append(param_details)
    
    # Creating a DataFrame for better visualization
    summary_df = pd.DataFrame(summary)
    
    # Adding total summary row
    total_summary = pd.DataFrame([{
        "Module": "Total",
        "Shape": "",
        "Num Parameters": total_params,
        "Trainable": trainable_params,
    }])
    
    summary_df = pd.concat([summary_df, total_summary], ignore_index=True)
        
    return summary_df

def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of total parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module) -> float:
    """
    Calculate the size of a PyTorch model in megabytes (MB).

    This function iterates through the model's parameters and buffers,
    calculates their sizes in bytes, and then converts the total size to megabytes.

    Args:
        model (torch.nn.Module): The PyTorch model to calculate the size of.

    Returns:
        float: The size of the model in megabytes (MB).
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def compare_parameters(model: nn.Module, base_model: nn.Module, method: str  =None):
    """
    Compare the parameters of the model before and after fine-tuning.

    Args:
        model (nn.Module): The fine-tuned model.
        base_model (nn.Module): The base model before fine-tuning.
        method (str): The fine-tuning method used, e.g., 'lora', 'surgical'.
    """
    if method == "lora":
        params_before = dict(base_model.named_parameters())
        for name, param in model.base_model.named_parameters():
            if "lora" not in name:
                continue
            logging.info(
                f"New parameter {name:<13} | {param.numel():>5} parameters | updated"
            )

        for name, param in model.base_model.named_parameters():
            if "lora" in name:
                continue
            name_before = (
                name.partition(".")[-1]
                .replace("original_", "")
                .replace("module.", "")
                .replace("modules_to_save.default.", "")
                .replace("base_layer.", "")
            )
            param_before = params_before[name_before]
            if torch.allclose(param, param_before):
                logging.info(
                    f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated"
                )
            else:
                logging.info(
                    f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated"
                )
    else:
        params_before = dict(base_model.named_parameters())
        params_after = dict(model.named_parameters())

        for name in params_after.keys():
            if name not in params_before:
                logging.info(
                    f"New parameter {name:<13} | {params_after[name].numel():>5} parameters | added"
                )
                continue
            if torch.allclose(params_after[name], params_before[name]):
                logging.info(
                    f"Parameter {name:<13} | {params_after[name].numel():>7} parameters | not updated"
                )
            else:
                logging.info(
                    f"Parameter {name:<13} | {params_after[name].numel():>7} parameters | updated"
                )


def save_model(model: nn.Module, path: str):
    """
    Save the model weights to the specified path.

    Args:
        model (torch.nn.Module): The model.
        path (str): The path to save the model weights.
    """
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device, strict: bool = False):
    """
    Load the model weights from the specified path.

    Args:
        model (torch.nn.Module): The model.
        path (str): The path to load the model weights from.
        device (torch.device): The device to load the model to.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    state_dict = torch.load(path, map_location=device)
    model_state_dict = model.state_dict()

    matched_layers = {
        k: v
        for k, v in state_dict.items()
        if k in model_state_dict and v.size() == model_state_dict[k].size()
    }
    unmatched_layers = {
        k: v
        for k, v in state_dict.items()
        if k not in model_state_dict or v.size() != model_state_dict[k].size()
    }

    if matched_layers:
        model_state_dict.update(matched_layers)
        model.load_state_dict(model_state_dict, strict=strict)
        logging.info(
            f"Loaded matched layers from {path}: {list(matched_layers.keys())}"
        )
    else:
        logging.warning(f"No matched layers found in {path}")

    if unmatched_layers:
        logging.warning(f"Unmatched layers not loaded: {list(unmatched_layers.keys())}")

    return model
