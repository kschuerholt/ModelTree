import torch
import torchvision
import copy
import numpy as np
from torch import nn
from typing import Dict


def load_model() -> nn.Module:
    """
    Loads a Vision Transformer (ViT) model from PyTorch's torchvision library.

    Returns:
        nn.Module: The loaded Vision Transformer model.
    """
    model = torchvision.models.vit_b_16()
    return model


def get_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Retrieves the state dictionary of the given model.

    Args:
        model (nn.Module): The model from which to retrieve the state dictionary.

    Returns:
        Dict[str, torch.Tensor]: The state dictionary of the model.
    """
    return model.state_dict()


def add_noise_to_state_dict(
    state_dict: Dict[str, torch.Tensor], noise_level: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Adds Gaussian noise to the state dictionary of a model.

    Args:
        state_dict (Dict[str, torch.Tensor]): The original state dictionary.
        noise_level (float): The standard deviation of the Gaussian noise to be added.

    Returns:
        Dict[str, torch.Tensor]: The state dictionary with added noise.
    """
    noisy_state_dict = copy.deepcopy(state_dict)
    for key in noisy_state_dict.keys():
        noisy_state_dict[key] += (
            noise_level
            * torch.randn(noisy_state_dict[key].shape)
            * noisy_state_dict[key]
        )
    return noisy_state_dict


def are_dicts_close(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor], key: str
) -> bool:
    """
    Checks if two tensors in two state dictionaries are approximately equal.

    Args:
        dict1 (Dict[str, torch.Tensor]): The first state dictionary.
        dict2 (Dict[str, torch.Tensor]): The second state dictionary.
        key (str): The key of the tensor to compare in both state dictionaries.

    Returns:
        bool: True if the tensors are approximately equal, False otherwise.
    """
    return torch.allclose(dict1[key], dict2[key])
