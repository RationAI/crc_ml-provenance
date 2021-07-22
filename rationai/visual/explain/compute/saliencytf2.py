"""
Saliency computation engine.

Wraps tf-keras-vis methods in an interface more consistent with the other
methods (SaliencyVis).
"""
import numpy as np
import tensorflow.keras.backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize as norm


def get_grad_modifier(modifier_name):
    """
    Get modifier function for obtained gradients.

    Parameters
    ----------
    modifier_name : str or None
        The identifier of the modifier function.
        Potential values are: 'absolute', 'relu'  and None.

    Returns
    -------
    Callable
        A function to modify gradients with. If None was passed, an identify
        function.
    """
    if modifier_name is None:
        return lambda grads: grads
    elif modifier_name == 'absolute':
        return lambda grads: K.abs(grads)
    elif modifier_name == 'relu':
        return lambda grads: K.relu(grads)

    raise ValueError('unknown gradient modifier: {}'.format(modifier_name))


def sal_loss(output):
    """
    The loss against which to compute the gradients for saliency.
    """
    # [0] _ -> batch
    # _ [0] -> bin val for cancer detect
    # TODO: choose class we want to focus on
    return output[0][0]


class SaliencyVis:
    """
    Wrapper class for saliency computation.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model: keras Model
            The model to be explained
        """
        self.saliency = Saliency(model, model_modifier=None, clone=False)

    def visualize_saliency(
            self,
            seed_input,
            loss,
            gradient_modifier=None,
            keep_dims=False,
            normalize=False
    ):
        """
        Compute the saliency heatmap.

        Parameters
        ----------
        seed_input : np.array
            The input image.
        loss : callable
            The loss function against which to compute the saliency map.
        gradient_modifier : callable
            A function over the gradients, to modify the resulting map.
        keep_dims : bool
            Whether to keep the original dimensions of the image.
        normalize : bool
            Whether to normalize the output.
        """
        saliency_map = self.saliency(
            loss,
            seed_input,
            gradient_modifier=gradient_modifier,
            keepdims=keep_dims
        )

        if normalize:
            saliency_map = norm(saliency_map)

        if keep_dims:
            saliency_map = np.array(saliency_map)

        return saliency_map[0]
