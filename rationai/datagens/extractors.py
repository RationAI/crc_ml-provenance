from __future__ import annotations

import abc
import cv2
import numpy as np
import openslide
from PIL import Image
from pathlib import Path
from pandas import DataFrame

from nptyping import NDArray
from typing import (
    NoReturn,
    Optional,
    Tuple,
    Type
)

from rationai.datagens.augmenters import AugmentInterface
from rationai.utils import (
    DirStructure,
    join_module_path,
    load_from_module
)


def load_extractor(identifier: dict) -> Type[SlideExtractor]:
    """Returns an Extractor instance.

    Args:
        identifier: dict
            'class_name': string - Name of a class to initialize
             'config': dict - A dict with parameters is passed to __init__()
    """

    class_name = identifier['class_name']
    config = identifier.get('config', dict())

    path = join_module_path(__name__, class_name)
    if path is None:
        raise ValueError(f'Invalid Extractor identifier: {class_name} not found')
    return load_from_module(path, **config)


class SlideExtractor(abc.ABC):
    """Base class for tile extraction from whole slide images.

    Arguments are provided by Datagen class.

    Args:
        config: dict - Entire config file is provided by Datagen.
        dir_struct: DirStructure - Access to paths.
        use_augment: bool - Whether to perform augmentation.
    """

    def __init__(self, config: dict, dir_struct: DirStructure, use_augment: bool):
        self.config = config
        self.dir_struct = dir_struct
        self.use_augment = use_augment

        self.img_size = tuple(self.config['model']['input_shape'][:2])
        self.slide_level = (self.config['data']['meta']['level'])

    @abc.abstractmethod
    def __call__(self, pddf: Type[DataFrame]) -> Tuple[NDArray, NDArray]:
        """Implementation should return an extracted batch."""
        raise NotImplementedError("Extractor's __call__ method not implemented.")

    def set_augmenter(self, aug: Type[AugmentInterface]) -> NoReturn:
        """Sets an augmenter instance"""
        self.augmenter = aug

    def open_slide(self, path: Path) -> Type[openslide.OpenSlide]:
        """Returns an opened whole slide image."""
        slide_fn = str(path.resolve())
        return openslide.open_slide(slide_fn)

    def extract_tile(self,
                     os_slide: Type[openslide.OpenSLide],
                     coords: Tuple[int, int],
                     convert: Optional[str] = None) -> NDArray:
        """Returns a tile extracted from a whole slide image.

        Args:
            os_slide : OpenSlide
                Whole slide image handle.

            coords : tuple(float, float)
                Tile upper left corner at level 0.

            convert : string
                Parameter for PIL.Image.convert() method.

        Returns:
            A tile as a numpy array.
        """
        bg_tile = Image.new('RGB', self.img_size, '#FFFFFF')
        im_tile = os_slide.read_region(
            location=coords, level=self.slide_level, size=self.img_size)
        bg_tile.paste(im_tile, None, im_tile)

        if convert:
            bg_tile = bg_tile.convert(convert)

        return np.array(bg_tile)


class BinaryClassExtractor(SlideExtractor):
    """Whole slide image extractor for binary classification.

    Coordinate map schema notes:
        Required columns: 'slide_name', 'coord_x', 'coord_y'
        Label column can be configured.
    """

    def __init__(self,
                 config: dict,
                 dir_struct: DirStructure,
                 use_augment: bool):
        super().__init__(config, dir_struct, use_augment)
        self.slide_dir = dir_struct.get('input')
        self.label_col = self.config['data']['dirs']['dataset']['pd_label_col']

    def preprocess_input(self, image: NDArray) -> NDArray:
        """Returns scales and normalized ndarray."""
        return (image / 127.5) - 1

    def __call__(self, pddf: Type[DataFrame]) -> Tuple[NDArray, NDArray]:
        """Returns extracted batch.

        Returns:
            Pair of ndarrays (tiles, binary labels)
        """
        result = {'image': [], 'class': []}

        for _, row in pddf.iterrows():
            slide_name = Path(row['slide_name'])
            coords = (row['coord_x'], row['coord_y'])
            label = row[self.label_col]

            os_slide = self.open_slide(self.slide_dir / slide_name.with_suffix('.mrxs'))
            img = self.extract_tile(os_slide, coords)
            os_slide.close()

            result['class'].append(label)
            result['image'].append(
                self.preprocess_input(
                    cv2.resize(
                        src=self.augmenter(img) if self.use_augment else img,
                        dsize=self.img_size,
                        interpolation=cv2.INTER_LINEAR
                    )
                )
            )

        return np.array(result['image']), np.array(result['class'])


class SegmentationExtractor(SlideExtractor):
    """Image segmentation extractor class.

    Coordinate map schema notes:
        Required columns: 'slide_id', 'x', 'y'
    """
    def __init__(self, config: dict, dir_struct: DirStructure, use_augment: bool):
        super().__init__(config, dir_struct, use_augment)

        # RGB feature vectors (.tif)
        self.rgb_dir = dir_struct.get('input')
        # B&W labels - binary masks (.tif)
        self.label_dir = dir_struct.get('label')

    def __call__(self, pddf: DataFrame) -> Tuple[NDArray, NDArray]:
        """Returns extracted batch.

        Returns:
            Pairs of ndarrays (RGB tile, binary mask)"""
        result = {'image': [], 'label': []}

        for _, row in pddf.iterrows():
            coordinates = (row['x'], row['y'])
            slide_id = Path(row['slide_id'])
            rgb_tif = self.open_slide(self.rgb_dir / slide_id.with_suffix('.tif'))
            label_tif = self.open_slide(self.label_dir / slide_id.with_suffix('.tif'))

            rgb_im = self.extract_tile(rgb_tif, coordinates)
            # expand grayscale img dims to contain a single channel (W x H -> W x H x C)
            label_im = np.expand_dims(self.extract_tile(label_tif, coordinates, convert='L'), -1)
            rgb_tif.close()
            label_tif.close()

            result['image'].append(rgb_im)
            result['label'].append(label_im)

        X = np.array(result['image'])
        y = np.array(result['label'])

        if self.use_augment:
            X, y = self.augmenter(images=X, segmentation_maps=y)

        X = (X / 127.5) - 1
        y = y // 255.0

        return X, y.reshape((len(pddf), -1, 1)).astype(np.float32)
