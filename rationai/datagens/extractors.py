# Standard Imports
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from typing import Any
from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.lib.index_tricks import nd_grid

from numpy.lib.shape_base import tile

# Third-party Imports
import openslide
from openslide import OpenSlide
import numpy as np
from nptyping import NDArray
from PIL import Image

# Local Imports
from rationai.datagens.augmenters import Augmenter
from rationai.datagens.samplers import SampledEntry

class Extractor(ABC):

    @abstractmethod
    def __call__(self):
        """Process sampled entries into valid network input (and output)"""

class OpenslideExtractor(Extractor):
    def __init__(self, augmenter: Augmenter, threshold: float):
        self.augmenter = augmenter
        self.threshold = threshold

    def __call__(self, sampled_entries: List[SampledEntry]) -> Tuple[NDArray, NDArray]:
        """Converts entries into network input/label tuple.

        Args:
            entries (List[dict]): Entries from a DataFrame

        Returns:
            Tuple[NDArray, NDArray]: Network inputs and labels.
        """
        inputs, labels = [], []
        for sampled_entry in sampled_entries:
            x, y = self.__process_entry(sampled_entry)
            x, y = self.__augment_input(x, y)
            x, y = self.__normalize_input(x, y)
            inputs.append(x)
            labels.append(y)
        return np.array(inputs), np.array(labels)

    def __process_entry(self, sampled_entry: SampledEntry) -> Tuple[NDArray, NDArray]:
        """Extracts a tile from a slide at coordinates specified by the parsed entry.

        Args:
            parsed_entry (OpenslideExtractor.ParsedEntry): Parsed entry

        Returns:
            Tuple[NDArray, NDArray]: Input/label tuple
        """
        wsi = self.__open_slide(sampled_entry.data['slide_name'])
        x = self.__extract_tile(wsi,
            (sampled_entry.entry['coord_x'], sampled_entry.entry['coord_y']),
            sampled_entry.metadata['tile_size'],
            sampled_entry.metadata['sampled_level']
        )
        y = sampled_entry.entry['center_tumor_tile'] > self.threshold
        wsi.close()
        return x, y

    def __open_slide(self, slide_name: str) -> OpenSlide:
        """Opens slide witha given name in `slide_dir` directory.

        Args:
            slide_name (str): Name of a slide.

        Returns:
            OpenSlide: File handler to WSI
        """
        slide_fp = str((self.slide_dir / slide_name).with_suffix('.mrxs'))
        wsi = openslide.open_slide(slide_fp)
        return wsi

    def __extract_tile(
        self,
        wsi: OpenSlide,
        coords: Tuple[int, int],
        tile_size: int,
        level: int) -> NDArray:
        """Extracts a tile from a slide using the supplied coordinate values.

        Args:
            wsi (OpenSlide): File handler to WSI
            coords (Tuple[int, int]): (x,y) coordinates of a tile to be extracted
                at OpenSlide level 0 resolution.
            tile_size (int): Size of the tile to be extracted.
            level (int): Resolution level from which tile should be extracted.

        Returns:
            NDArray: RGB Tile represented as numpy array.
        """
        bg_tile = Image.new('RGB', self.tile_size, '#FFFFFF')
        im_tile = wsi.read_region(
            location=coords, level=level, size=tile_size
        )
        bg_tile.paste(im_tile, None, im_tile)
        return np.array(bg_tile)

    def __normalize_input(self, x: NDArray, y: NDArray) -> Tuple[NDArray]:
        """Normalizes images pixel values from [0-255] to [0-1] range.

        TODO: Maybe make this part of an augmenter?

        Args:
            x (NDArray): Network input
            y (NDArray): Label

        Returns:
            Tuple[NDArray]: Normalized input/label pair.
        """
        x = (x / 127.5) - 1
        return x, y

    def __augment_input(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        """Applies augmentation on the input/label pair.

        Args:
            x (NDArray): Network input
            y (NDArray): Label

        Returns:
            Tuple[NDArray, NDArray]: Augmented input/label pair.
        """
        return self.augmenter(x, y)