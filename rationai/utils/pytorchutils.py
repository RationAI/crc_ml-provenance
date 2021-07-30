"""
Utility functions for working with PyTorch objects.
"""
from typing import Optional

import torch

from rationai.utils.typealias import TorchOptimGenerator, TorchRegularizer


def get_pytorch_loss(name: str) -> Optional[torch.nn.modules.loss._Loss]:
    """
    Resolve PyTorch loss.

    Parameters
    ----------
    name : str
        The name of the loss to be used. Currently available are:
            'BinaryCrossentropy'.
        If anything other than this is provided, None is returned.

    Return
    ------
    Optional[torch.nn.modules._Loss]
        A PyTorch loss function.
    """
    loss = None

    try:
        if name.lower() == 'binarycrossentropy':
            loss = torch.nn.BCELoss()
    except AttributeError:
        # `name` is not a string
        pass

    return loss


def get_pytorch_optimizer(name: str, config: dict) -> Optional[TorchOptimGenerator]:
    """
    Resolve PyTorch optimizer.

    Resulting function should be used by calling it on a network's
    .parameters() to create an optimizer over the network.

    Parameters
    ----------
    name : str
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
    Optional[TorchOptimGenerator]
        A function over a network's parameters to create the final
        Optimizer.

    Raise
    -----
    ValueError
        When invalid `config` is passed as a parameter (see above).
    """
    if name is None:
        return None

    # RMSProp optimizer
    if name.lower() == 'rmsprop':
        try:
            lr: float = config['lr']
            eps: float = config['epsilon']
            weight_decay: float = config['rho']
            momentum: float = config['momentum']
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
        # Fallback to None
        optimizer = None

    return optimizer


def get_pytorch_regularizer(name: str, config: dict) -> Optional[TorchRegularizer]:
    """
    Resolve PyTorch regularization method.

    Resulting function should be used by iterating for all p in model.params()
    and summing over the individual items.

    Parameters
    ----------
    name : str
        The name of the regularizer to be used. Currently available are:
            'L1', 'L2'.
        If anything other than these is provided, None is returned.
    config : dict
        A dictionary of parameters necessary for regularization initialization:
            - 'L1': {'l1': lambda weight coefficient (e.g. 0.1)}
            - 'L2': {'l2': lambda weight coefficient (e.g. 0.1)}

    Return
    ------
    Optional[TorchRegularizer]
        The regularization function to be applied on network parameters.

    Raise
    -----
    ValueError
        When invalid `config` is passed as a parameter (see above).
    """
    if name is None:
        return None

    # L1 regularization
    if name.lower() == 'l1':
        try:
            weight: float = config['l1']
        except (KeyError, TypeError):
            raise ValueError('missing proper config for L1 regularization')
        else:
            def regularizer(model_params):
                return torch.norm(model_params, 1) * weight

    # L2 regularization
    elif name.lower() == 'l2':
        try:
            weight: float = config['l2']
        except (KeyError, TypeError):
            raise ValueError('missing proper config for L2 regularization')
        else:
            def regularizer(model_params):
                return (model_params ** 2).sum() * weight
    else:
        # Fallback to None
        regularizer = None

    return regularizer
