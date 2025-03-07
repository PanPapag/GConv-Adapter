import copy
import torch.nn as nn
from typing import List


class BitFit:
    def __init__(self, model: nn.Module, target_modules: List[str] = None):
        """
        Initialize the BitFit class as described in the paper https://arxiv.org/pdf/2106.10199

        Args:
            model (nn.Module): The model to be fine-tuned.
            target_modules (list, optional): List of target modules to apply BitFit to. If None, defaults to all linear layers' bias terms.
        """
        self.model = copy.deepcopy(model)
        self.target_modules = (
            target_modules if target_modules is not None else self.get_target_modules()
        )
        self._configure_layers()

    def get_target_modules(self) -> List[str]:
        """
        Get the target modules for BitFit.

        Returns:
            list: List of target modules.
        """
        target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name + ".bias")
        return target_modules

    def _configure_layers(self):
        """
        Configure which layers to train based on the provided options.
        """
        for name, param in self.model.named_parameters():
            if name not in self.target_modules:
                param.requires_grad = False

    def apply(self) -> nn.Module:
        """
        Apply BitFit to the model.

        Returns:
            nn.Module: The model with BitFit applied.
        """
        self._configure_layers()
        return self.model
