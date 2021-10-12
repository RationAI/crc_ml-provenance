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

from rationai.utils.config import ConfigProto


class AugmentInterface(abc.ABC):
    """Interface for data augmentation.

    If augmentation is turned on, an extractor calls its __call__ method after tile extraction.
    """

    @abc.abstractmethod
    def __call__(self, **kwargs):
        """Called by an extractor to perform data augmentation."""
        raise NotImplementedError('Override __call__ method to perform a data transformation')


class ImgAugAugmenter(AugmentInterface):
    """Uses image augmenter: imgaug.augmenters.iaa

    This class should not be used directly, but subclassed.

    Attributes
    ----------
    augmenter : imgaug.augmenters.meta.Augmenter
        The augmenter containing image transformation configuration, used when the __call__ method is used.
    """
    def __init__(self):
        self.augmenter = iaa.Noop()

    def __call__(self, **kwargs):
        """Augments input image(s).

        Return
        ------
        # TODO: Check the return type
        """
        return self.augmenter.augment(**kwargs)


class ImageAugmenter(ImgAugAugmenter):
    """
    For information about what augmentations are used, see `ImageAugmenterConfig` class.

    Attributes
    ----------
    augmenter : imgaug.augmenters.Sequential
        The augmenter containing image transformation configuration, used when the __call__ method is used.
    """

    def __init__(self, config: ImageAugmenterConfig):
        super().__init__()
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


class NoOpImageAugmenter(ImgAugAugmenter):
    """This is the class to be used when no image augmentation operation is to be done."""
    pass


# Config classes

@dataclass
class ImageAugmenterConfig(ConfigProto):
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
    def parse(cls, config_path: Path) -> ImageAugmenterConfig:
        """
        Parse SlideAugmenter configuration from a JSON file.

        Parameters
        ----------
        config_path : pathlib.Path
            The path to the configuration JSON file to parse.

        Return
        ------
        ImageAugmenterConfig
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
