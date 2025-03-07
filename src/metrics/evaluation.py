import torch
from typing import Tuple


def accuracy(preds: torch.Tensor, labels: torch.Tensor):
    """
    Calculate the accuracy of predictions.

    Args:
        preds (torch.Tensor): The predicted labels.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        float: The accuracy of the predictions.
    """
    return (preds == labels).sum().item() / len(labels)


def calculate_loss_and_accuracy(output: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate the loss and accuracy for the given output, mask, and labels.

    Args:
        output (torch.Tensor): The model output.
        mask (torch.Tensor): The mask indicating which nodes are included in the calculation.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        tuple: A tuple containing the loss and accuracy.
    """
    loss = torch.nn.functional.nll_loss(output[mask], labels[mask])
    acc = accuracy(output[mask].argmax(dim=1), labels[mask])
    return loss, acc
