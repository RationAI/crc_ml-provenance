"""
Transformations to models.
"""
import tensorflow.keras.activations as tf_activations
from tensorflow.python.framework.ops import Tensor
# from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from typing import Callable

# function (Tensor) -> Tensor
Activation = Callable[[Tensor], Tensor]


def replace_activation(
        model: Model,
        layer_idx: int = -1,
        activation: Activation = tf_activations.linear) -> Model:
    """
    Replaces activation function of the specified layer.

    Parameters
    ----------
    model : keras Model
        The model whose last layer to replace.

    Returns
    -------
    keras Model
        A new model, with replaced activation function for layer layer_idx.
    """
    model.layers[layer_idx].activation = activation
    return model

    # # TODO: just change activation to linear function
    # penultimate = model.layers[-2].output
    # new_final = Dense(1, name='Output_Dense_NoSigmoid')(penultimate)
    # new_model = Model(model.inputs, new_final)
    # new_model.get_layer('Output_Dense_NoSigmoid').set_weights(
    #     model.layers[-1].get_weights()
    # )
    # return new_model
