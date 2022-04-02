
import torch.nn as nn
from collections import OrderedDict


class GradCam(nn.Module):
    """
    This Module performs grad cam computation for a model consisting of two Modules applied in order.
    To use it on an arbitrary model, the model has to be manually dissected first.
    The dissection for a convolutional network is usually performed in such a way that the convolutional layers are represented
    by the first Module and the dense layers are left in the second Module.
    
    The dissection can be arbitrary but it is expected that the output of the first module has a shape of [N, C, W, H],
    meaning it has to be a batch of size N, each element containing C feature maps of shape W x H (2D matrices)
    so that the spatial information can be transformed into a heatmap overlay for the input picture.
    Also, the last activation layer containing Sigmoids, SoftMAX or other stuff should be stripped away either.
    """
    def __init__(self, model_before_cam: nn.Module, model_after_cam: nn.Module, *args, **kwargs):
        """Initialize the GradCam for a dissected model, split into two parts.
        The parts are expected to split at the layer of interest and together have to make the full model.

        Args:
            model_before_cam (nn.Module): Part of the inspected model containing up to the convolutional layer we want to inspect 
            model_after_cam (nn.Module): Remaining layers of the inspected model
        """
        super().__init__(*args, **kwargs)
        self.model_before_cam = model_before_cam
        self.model_after_cam = model_after_cam
        
        self.gradients = None
        self.activations = None
    
    # hook for storing the gradients of the activations
    def store_grads_hook(self, grad):
        self.gradients = grad.detach().clone()
        
    def forward(self, x):
        # remember the activation
        activations = self.model_before_cam(x)
        self.activations = activations.detach().clone()
        #print("Detached A:", self.activations)
        
        # register the backward hook (we dont need the reference to it)
        _ = activations.register_hook(self.store_grads_hook)
        
        # apply the remaining layers
        return self.model_after_cam(activations)
        
    # getter fro the stored gradients
    def get_activations_gradient(self):
        return self.gradients
    
    # getter for the stored activations
    def get_activations(self):
        return self.activations
