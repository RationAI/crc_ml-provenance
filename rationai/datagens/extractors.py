# Standard Imports
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from typing import Any
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

class Extractor(ABC):

    @abstractmethod
    def __call__(self):
        """Process sampled entry into valid network input (and output)"""

    @abstractmethod
    def __parse_entry(self):
        """Parse sampled entry"""

    @abstractmethod
    def __process_entry(self):
        """Process parsed entry into valid network input (and output)"""

class OpenslideExtractor(Extractor):
    def __init__(self, slide_dir: Path, tile_size: int, sample_level: int, augmenter: Augmenter):

        @dataclass
        class ParsedEntry:
            slide_name: str
            coords: Tuple[int, int]
            label: bool

        self.slide_dir = slide_dir
        self.tile_size = tile_size
        self.augmenter = augmenter
        self.sample_level = sample_level

    def __call__(self, entries: List[dict]) -> Tuple[NDArray, NDArray]:
        """Converts a single entry into network input/label tuple.

        Args:
            entries (List[dict]): Entries from a DataFrame

        Returns:
            Tuple[NDArray, NDArray]: Network input/Label tuple
        """
        for entry in entries:
            parsed_entry = self.__parse_entry(entry)
            x, y = self.__process_entry(parsed_entry)
            x, y = self.__augment_input(x, y)
            x, y = self.__normalize_input(x, y)
            return x, y

    def __parse_entry(self, entry: dict) -> OpenslideExtractor.ParsedEntry:
        """Parses entry from DataFrame.

        Args:
            entry (dict): Single entry

        Returns:
            OpenslideExtractor.ParsedEntry: Parsed entry
        """
        slide_name = entry['slide_name']
        coords = (entry['coord_x'], entry['coord_y'])
        label = entry['label']
        return self.ParsedEntry(slide_name, coords, label)

    def __process_entry(self, parsed_entry: OpenslideExtractor.ParsedEntry) -> Tuple[NDArray, NDArray]:
        """Extracts a tile from a slide at coordinates specified by the parsed entry.

        Args:
            parsed_entry (OpenslideExtractor.ParsedEntry): Parsed entry

        Returns:
            Tuple[NDArray, NDArray]: Input/label tuple
        """
        wsi = self.__open_slide(parsed_entry.slide_name)
        x = self.__extract_tile(wsi, parsed_entry.coords)
        return x, parsed_entry.label

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

    def __extract_tile(self, wsi: OpenSlide, coords: Tuple[int, int]) -> NDArray:
        """Extracts an image from a slide using the supplied coordinate values.

        Args:
            wsi (OpenSlide): File handler to WSI
            coords (Tuple[int, int]): (x,y) coordinates of a tile to be extracted
                                            at OpenSlide level 0 resolution.

        Returns:
            NDArray: Extracted tile
        """
        bg_tile = Image.new('RGB', self.tile_size, '#FFFFFF')
        im_tile = wsi.read_region(
            location=coords, level=self.sample_level, size=self.tile_size
        )
        bg_tile.paste(im_tile, None, im_tile)
        return np.array(bg_tile)

    def __normalize_input(self, x: NDArray, y: NDArray) -> Tuple[NDArray]:
        """Normalizes images pixel values from [0-255] to [0-1] range.

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