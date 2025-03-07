import copy
import torch.nn as nn
from typing import List

class SurgicalFinetuning:
    def __init__(self, model: nn.Module, train_last_layer: bool = False, train_layer_norm: bool = False, train_batch_norm: bool = False, target_modules: List[str] = None):
        """
        Initialize the SurgicalFinetuning class.

        Args:
            model (nn.Module): The model to be fine-tuned.
            train_last_layer (bool): Whether to train only the last layer.
            train_layer_norm (bool): Whether to train LayerNorm layers.
            train_batch_norm (bool): Whether to train BatchNorm layers.
            target_modules (List[str], optional): List of target modules to be fine-tuned.
        """
        self.model = copy.deepcopy(model)
        self.train_last_layer = train_last_layer
        self.train_target_modules = target_modules
        self.train_layer_norm = train_layer_norm
        self.train_batch_norm = train_batch_norm
        self._configure_layers()

    def _configure_layers(self):
        """
        Configure which layers to train based on the provided options.
        """
        last_layer_name = self._get_last_layer_name() if self.train_last_layer else None

        for name, module in self.model.named_modules():
            for _, param in module.named_parameters(recurse=False):
                if self.train_last_layer and last_layer_name in name:
                    param.requires_grad = True
                elif self.train_layer_norm and isinstance(module, nn.LayerNorm):
                    param.requires_grad = True
                elif self.train_batch_norm and (isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d)):
                    param.requires_grad = True
                elif self.train_target_modules and any(target in name for target in self.train_target_modules):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def _get_last_layer_name(self) -> str:
        """
        Get the name of the last layer's weight in the model.

        Returns:
            str: The name of the last linear layer's weight.
        """
        last_layer_name = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                last_layer_name = name
                break
        return last_layer_name

    def apply(self) -> nn.Module:
        """
        Apply the configured fine-tuning settings to the model.

        Returns:
            nn.Module: The model with updated layer configurations.
        """
        self._configure_layers()
        return self.model
