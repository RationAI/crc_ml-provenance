"""
Utility functions for working with PyTorch objects.
"""
from typing import Callable, Optional

import torch


def get_pytorch_regularizer(
        regularizer_name: str, config: dict
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """
    Resolve PyTorch regularization method.

    Resulting function should be used by iterating for all p in model.params()
    and summing over the individual items.

    Parameters
    ----------
    regularizer_name : str
        The name of the regularizer to be used. Currently available are:
            'L1', 'L2'.
        If anything other than these is provided, None is returned.
    config : dict
        A dictionary of parameters necessary for regularization initialization:
            - 'L1': {'l1': lambda weight coefficient (e.g. 0.1)}
            - 'L2': {'l2': lambda weight coefficient (e.g. 0.1)}

    Return
    ------
    (torch.Tensor) -> torch.Tensor
        The regularization function to be applied on network parameters.

    Raise
    -----
    ValueError
        When invalid `config` is passed as a parameter (see above).
    """
    if regularizer_name is None:
        return None

    # L1 regularization
    if regularizer_name.lower() == 'l1':
        try:
            weight = config['l1']
        except (KeyError, TypeError):
            raise ValueError('missing proper config for L1 regularization')
        else:
            def regularizer(model_params):
                return torch.norm(model_params, 1) * weight

    # L2 regularization
    elif regularizer_name.lower() == 'l2':
        try:
            weight = config['l2']
        except (KeyError, TypeError):
            raise ValueError('missing proper config for L2 regularization')
        else:
            def regularizer(model_params):
                return (model_params ** 2).sum() * weight
    else:
        # Fallback to None
        regularizer = None

    return regularizer
