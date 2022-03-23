from operator import irshift
from numpy import isin
import torch
import torch.nn as nn
import torch.functional as F

from collections import OrderedDict

from typing import Union

#import logging
#log = logging.getLogger('gradcam')


def _load_params(model, source_state_dict: Union[str, dict]):    
    print("Loading weights from a supplied state_dict...")
    if isinstance(source_state_dict, str):
        source_state_dict = torch.load(source_state_dict)
    elif isinstance(source_state_dict, dict):
        source_state_dict = source_state_dict
    else:
        raise TypeError('The source state dict can only be string or a dictionary, not %s')
    # get a deep copy of the existing parameters so that we have a dictionary that matches the model parameters perfectly
    new_weights = model.state_dict()
    
    _missing_keys = []
    _loaded_params = []
    _unexpected_keys = set(source_state_dict.keys())
    # iterate over the parameters of this model
    for params_name in new_weights.keys():
        # if the parameter matches a parameter in the input state_dict, load it and not it
        if params_name in source_state_dict:
            new_weights[params_name] = source_state_dict[params_name]
            _loaded_params.append(params_name)
            _unexpected_keys.remove(params_name)
        else:
            _missing_keys.append(params_name)
    #load the updated weights
    model.load_state_dict(new_weights)
    print("Weights loaded:", _loaded_params)
    print("Weights not expected:", _unexpected_keys)
    print("Weights not present:", _missing_keys)



def dissect_sequential_model(sequential: nn.Sequential, layer_index: int):
    """
    Splits a torch.nn.Sequential model into two parts at a particular layer index

    Args:
        sequential (nn.Sequential): The input sequential model
        layer_index (int): The index of the layer at which the model is split.
                           The first module will contain layers up to the layer of the specified index (excluding the layer at that index)
                           and the second module will contain all the remaining layers (including the one at the specified index)

    Returns:
        (nn.Sequential, nn.Sequential): Tuple of the two sequential modules obtained by dissecting the input module.
    """
    _modules = list(sequential._modules.items())
    print(f'Model dissected after layer "{_modules[layer_index-1][0]}".')
    first_half = nn.Sequential(OrderedDict(_modules[:layer_index]))
    second_half = nn.Sequential(OrderedDict(_modules[layer_index:]))

    return first_half, second_half
      
      