from __future__ import annotations

import abc
import imgaug.augmenters as iaa
from typing import Type

from rationai.utils import join_module_path
from rationai.utils import load_from_module


def load_augmenter(identifier: dict) -> Type[AugmentInterface]:
    """Retrieves an Augmenter instance from the config

    Args:
        identifier : dict
            `class_name`: string - A name of a class to initialize
            `config`: dict - A dict with parameters is passed to __init__()
    """

    class_name = identifier['class_name']
    config = identifier.get('config', dict())

    path = join_module_path(__name__, class_name)
    if path is None:
        ValueError(f'Invalid Augmenter identifier: {class_name} not found.')
    return load_from_module(path, config)


class AugmentInterface(abc.ABC):
    """Interface for data augmentation.

    If augmentation is turned on,
    an extractor calls its __call__ method after tile extraction."""

    def __init__(self, config: dict):
        ...

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Called by an extractor before it sends the data to learning process"""
        raise NotImplementedError('Override __call__ method to perform a data transformation')


class SlideAugmenter(AugmentInterface):
    """Image augmenter: imgaug.augmenters.iaa"""
    def __init__(self, config: dict):
        self.augmenter = iaa.Sequential(
            [
                iaa.Fliplr(config.get('horizontal', 0), name='horizontal'),
                iaa.Flipud(config.get('vertical', 0), name='vertical'),
                iaa.AddToBrightness(
                    add=tuple(config.get('brightness', [0, 0])),
                    name='brightness'),
                iaa.AddToHueAndSaturation(
                    value_saturation=tuple(config.get('saturation', [0, 0])),
                    value_hue=tuple(config.get('hue', [0, 0])),
                    name='hue_and_saturation'),
                iaa.GammaContrast(
                    gamma=tuple(config.get('contrast', [0, 0])),
                    per_channel=False,
                    name='contrast'),
                iaa.geometric.Rot90(k=[0, 1], name='rotate90')
            ])

    def __call__(self, *args, **kwargs):
        """Augments input image(s) using flips, rotations,
        and perturbations of brightness, saturation, hue, and contrast.

        Single image arg calls:
            iaa.Sequential.augment_image(img)

        Otherwise *args and *kwargs are passed:
            iaa.Sequential(*args, **kwargs)

        The second option accepts two images thus,
        is suitable for segmentation use case to perform
        the same augmentation on pairs of images.
        """
        if len(args) == 1 and len(kwargs) == 0:
            return self.augmenter.augment_image(*args)
        return self.augmenter(*args, **kwargs)
