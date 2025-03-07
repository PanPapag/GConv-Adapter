import copy
import torch
import torch.nn as nn
from typing import List, Union

class LoRALayer(torch.nn.Module):
    """
    A Layer for Low-Rank Adaptation (LoRA) in neural networks.

    Args:
        in_dim (int): The input dimensionality.
        out_dim (int): The output dimensionality.
        rank (int): The rank for the low-rank adaptation.
        alpha (float): Scaling factor for the adaptation weights. If None, defaults to the rank value.
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: Union[int, float]):
        """
        Initialize the LoRALayer.

        Args:
            in_dim (int): The input dimensionality.
            out_dim (int): The output dimensionality.
            rank (int): The rank for the low-rank adaptation.
            alpha (float): Scaling factor for the adaptation weights. If None, defaults to the rank value.
        """
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha if alpha else rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRALayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the low-rank adaptation.
        """
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(nn.Module):
    """
    A Linear layer combined with a LoRALayer for low-rank adaptation.

    Args:
        linear (torch.nn.Linear): The base linear layer.
        rank (int): The rank for the low-rank adaptation.
        alpha (Union[int,float]): Scaling factor for the adaptation weights. If None, defaults to the rank value.
    """
    
    def __init__(self, linear: nn.Linear, rank: int, alpha: Union[int, float]):
        """
        Initialize the LinearWithLoRA layer.

        Args:
            linear (torch.nn.Linear): The base linear layer.
            rank (int): The rank for the low-rank adaptation.
            alpha (float): Scaling factor for the adaptation weights. If None, defaults to the rank value.
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LinearWithLoRA layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying both the base linear layer and the LoRALayer.
        """
        return self.linear(x) + self.lora(x)


class LoRA:
    """
    A class to apply Low-Rank Adaptation (LoRA) to a given model as described in the paper https://arxiv.org/pdf/2106.09685

    Attributes:
        model (nn.Module): The base model to which LoRA will be applied.
        rank (int): The rank of the low-rank adaptation.
        target_modules (list): List of module names to which LoRA will be applied.
    """

    def __init__(self, model: nn.Module, r: int, alpha: float = None, target_modules: List[str] = None):
        """
        Initialize the LoRA class.

        Args:
            model (nn.Module): The base model to which LoRA will be applied.
            r (int): The rank of the low-rank adaptation.
            alpha (float): Scaling factor of LoRA
            target_modules (list, optional): List of module names to which LoRA will be applied.
                                              If not provided, all non-bias Linear layers except the last one will be used.
        """
        self.model = copy.deepcopy(model)
        self.rank = r
        self.alpha = alpha
        self.target_modules = target_modules if target_modules is not None else self.get_target_modules()

    def get_target_modules(self) -> List[str]:
        """
        Identify all target modules in the model to which LoRA should be applied.

        Returns:
            List[str]: List of target module names.
        """
        target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)
        return target_modules

    def apply(self) -> nn.Module:
        """
        Apply LoRA to the target modules in the model.
        """
        for name, module in self.model.named_modules():
            if name in self.target_modules and isinstance(module, nn.Linear):
                # Split the full module name into its components
                module_names = name.split('.')
                submodule = self.model
                for sub_name in module_names[:-1]:
                    submodule = getattr(submodule, sub_name)
                
                # Get the old Linear layer to be replaced
                old_module = getattr(submodule, module_names[-1])
                
                # Create a new LinearWithLoRA layer
                new_module = LinearWithLoRA(old_module, rank=self.rank, alpha=self.alpha)
                
                # Replace the old module with the new module
                setattr(submodule, module_names[-1], new_module)
                        
        # Update parameters to ensure correct ones require gradients
        self.update_params()

        return self.model

    def update_params(self):
        """
        Update the parameters of the model to ensure only the correct ones require gradients.
        """
        for name, param in self.model.named_parameters():
            # Ensure only LoRA parameters require gradients
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
