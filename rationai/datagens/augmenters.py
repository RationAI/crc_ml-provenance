"""
Data augmenter definitions.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path

import imgaug
import imgaug.augmenters as iaa
import numpy as np

from rationai.utils.config import ConfigProto


class AugmentInterface(abc.ABC):
    """Interface for data augmentation.

    If augmentation is turned on, an extractor calls its __call__ method after tile extraction.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Called by an extractor to perform data augmentation."""
        raise NotImplementedError('Override __call__ method to perform a data transformation')


class SlideAugmenter(AugmentInterface):
    """Uses image augmenter: imgaug.augmenters.iaa

    For information about what augmentations are used, see `SlideAugmenterConfig` class.

    Attributes
    ----------
    augmenter : imgaug.augmenters.Sequential
        The augmenter containing image transformation configuration, used when the __call__ method is used.
    """
    def __init__(self, config: SlideAugmenterConfig):
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(config.horizontal_flip_proba, name='horizontal'),
            iaa.Flipud(config.vertical_flip_proba, name='vertical'),
            iaa.AddToBrightness(add=config.brightness_add_range, name='brightness'),
            iaa.AddToHueAndSaturation(
                value_saturation=config.saturation_add_range,
                value_hue=config.hue_add_range,
                name='hue_and_saturation'
            ),
            iaa.GammaContrast(gamma=config.contrast_scale_range, per_channel=False, name='contrast'),
            iaa.geometric.Rot90(k=config.rotate_90_deg_interval, name='rotate90')
        ])

    def __call__(self, *args, **kwargs) -> np.ndarray | imgaug.augmentables.batches.UnnormalizedBatch:
        """Augments input image(s) using flips, rotations, and changes in brightness, saturation, hue, and contrast.

        Single image arg calls:
            iaa.Sequential.augment_image(img)

        Otherwise *args and **kwargs are passed:
            iaa.Sequential(*args, **kwargs)

        The second option accepts two images thus, is suitable for segmentation use case to perform the same
        augmentation on pairs of images.

        Return
        ------
        # TODO: Check the return type
        """
        if len(args) == 1 and len(kwargs) == 0:
            return self.augmenter.augment_image(*args)
        return self.augmenter(*args, **kwargs)


# Config classes

@dataclass
class SlideAugmenterConfig(ConfigProto):
    # noinspection PyUnresolvedReferences
    """
    Configuration parser and wrapper for SlideAugmenter.

    Attributes
    ----------
    config_path : pathlib.Path
        The path to the parsed configuration file.
    horizontal_flip_proba : float
        Probability that an image will be flipped horizontally.
    vertical_flip_proba : float
        Probability that an image will be flipped vertically.
    brightness_add_range : tuple(float, float)
        Range from which to add to image brightness.
    saturation_add_range : tuple(float, float)
        Range from which to add to image saturation.
    hue_add_range : tuple(float, float)
        Range from which to add to image hue.
    contrast_scale_range : tuple(float, float)
        Range from which to choose scaling factor for contrast augmentation.
    rotate_90_deg_interval : tuple(int, int)
        Discrete interval from which to choose number of 90 degree rotations performed on an image.
    """
    config_path: Path
    horizontal_flip_proba: float = 0.
    vertical_flip_proba: float = 0.
    brightness_add_range: tuple[float, float] = (0., 0.)
    saturation_add_range: tuple[float, float] = (0., 0.)
    hue_add_range: tuple[float, float] = (0., 0.)
    contrast_scale_range: tuple[float, float] = (0., 0.)
    rotate_90_deg_interval: tuple[int, int] = (0, 1)

    @classmethod
    def parse(cls, config_path: Path) -> SlideAugmenterConfig:
        """
        Parse SlideAugmenter configuration from a JSON file.

        Parameters
        ----------
        config_path : pathlib.Path
            The path to the configuration JSON file to parse.

        Return
        ------
        SlideAugmenterConfig
            An initialized instance of the config wrapper.
        """
        with open(config_path, 'r') as file_input:
            config = json.load(file_input)

        augmenter_config = config['extractor']['augmenter']

        parsed_config = dict(
            config_path=config_path,
            horizontal_flip_proba=float(augmenter_config['horizontal_flip']),
            vertical_flip_proba=float(augmenter_config['vertical_flip']),
            brightness_add_range=tuple(augmenter_config['brightness_range']),
            saturation_add_range=tuple(augmenter_config['saturation_range']),
            hue_add_range=tuple(augmenter_config['hue_range']),
            contrast_scale_range=tuple(augmenter_config['contrast_range']),
            rotate_90_deg_interval=tuple(augmenter_config['rotate_interval'])
        )

        return cls(**parsed_config)
