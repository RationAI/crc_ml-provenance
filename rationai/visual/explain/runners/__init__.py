"""
Histopathology pipeline runners for the different methods.

`bcnasnet_*` files are to be used with the GLADOS-CHCK* family of models.
  (also known as macrotile models).
`pretrainedvgg16_*` files are to be used with the VGG16-TF2-DATASET* family of
  models. (also known as single tile models).

`*_base.py` files are to be used when a single set of non-overlapping
  explanations is to be obtained.
`*_blend.py` files are to be used when an average over all overlapping
  explanations is to be obtained.
"""

from .occlusion.pretrainedvgg16_blend import OcclusionRunner
from .saliency.pretrainedvgg16_blend import SaliencyRunner

__all__ = [
  'OcclusionRunner',
  'SaliencyRunner'
]
