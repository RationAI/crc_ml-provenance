"""
Utility functions for working with PyTorch objects.
"""
from typing import Callable, Optional, Iterator

import torch


def get_pytorch_optimizer(
        optimizer_name: str, config: dict
) -> Optional[Callable[[Iterator[torch.Tensor]], torch.optim.Optimizer]]:
    """
        Resolve PyTorch optimizer.

        Resulting function should be used by calling it on a network's
        .parameters() to create an optimizer over the network.

        Parameters
        ----------
        optimizer_name : str
            The name of the optimizer to be used. Currently available are:
                'RMSProp'.
            If anything other than this is provided, None is returned.
        config : dict
            A dictionary of parameters necessary for regularization initialization:
                - 'RMSProp': {
                    'lr': learning rate (e.g. 0.1),
                    'epsilon': term added to the denominator to improve
                        numerical stability (e.g. 1e-8),
                    'rho': L2 penalty (e.g. 0.01),
                    'momentum': momentum factor (e.g. 0.1)
                }

        Return
        ------
        (Iterator[torch.Tensor]) -> torch.optim.Optimizer
            A function over a network's parameters to create the final
            Optimizer.

        Raise
        -----
        ValueError
            When invalid `config` is passed as a parameter (see above).
        """
    if optimizer_name is None:
        return None

    # RMSProp optimizer
    if optimizer_name.lower() == 'rmsprop':
        try:
            lr = config['lr']
            eps = config['epsilon']
            weight_decay = config['rho']
            momentum = config['momentum']
        except (KeyError, TypeError):
            raise ValueError('missing proper config for RMSProp optimizer')
        else:
            def optimizer(params):
                return torch.optim.RMSprop(
                    params,
                    lr=lr,
                    eps=eps,
                    weight_decay=weight_decay,
                    momentum=momentum
                )
    else:
        optimizer = None

    return optimizer


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
