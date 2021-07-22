from __future__ import annotations

import argparse
import logging
import numpy as np
import openslide
import tqdm
import pandas as pd

from multiprocessing import Pool
from pathlib import Path
from PIL import Image
from time import time
from typing import (
    Generator,
    NoReturn,
    Optional,
    Type,
    Union
)

from rationai.data.utils import mkdir

# Type hints
OSlide = Type[openslide.OpenSlide]


class CreateMap:
    """Creates coord_maps & dataset for WSI segmentation.

    Computational parameters are automatically reflected
    in the directory name of coordinate maps and dataset file name.

    Expects the respective files to exist in following dirs:
        <data collection dir>
            rgb        # WSIs
            label      # binary masks (labels)
            bg         # binary masks (background masks)

    Creates:
        <data collection dir>/
            coord_maps/<dataset_name>/
            datasets/<dataset_name>
    """

    def __init__(self,
                 data_base_dir: Union[Path, str],
                 ds_prefix: str,
                 level: int,
                 tile_sile: int = 512,
                 step_size: int = 512,
                 min_tissue: Optional[int] = 0.5,
                 max_tissue: Optional[int] = None,
                 file_pattern: str = '*.tif',
                 max_workers: int = 4):

        self.level = level
        self.tile_size = tile_sile
        self.step_size = step_size
        self.file_pattern = file_pattern
        self.min_tissue = min_tissue
        self.max_tissue = max_tissue
        self.max_workers = max_workers

        self.log = logging.getLogger('CreateMap')
        self._init_dir_structure(Path(data_base_dir), ds_prefix)

    def run(self) -> NoReturn:
        """Performs segmentation data preprocessing"""
        self.log.info('Generating coord_maps')
        gen = self.iterate_inputs(file_pattern=self.file_pattern)

        if self.max_workers > 1:
            self.log.info(f'Multiprocessing ON: using pool of {self.max_workers} workers')
            with Pool(self.max_workers) as pool:
                data = list(gen)
                [_ for _ in tqdm.tqdm(pool.imap(self.create_coord_map, data),
                                      total=len(data))]
        else:
            self.log.info('Multiprocessing OFF')
            for entry in gen:
                self.create_coord_map(entry)

        self.log.info(f'Coord_maps saved to: "{self.coord_maps_dir}"')
        self.create_dataset()

    def create_coord_map(self, slide_obj_entry: dict) -> NoReturn:
        """Creates a coordinate map for a single slide."""
        slide_id = slide_obj_entry['rgb'].stem
        output_fn = (mkdir(self.coord_maps_dir, exist_ok=True) / slide_id).with_suffix('.gz')

        if output_fn.exists():
            self.log.debug(f'Coordinate map for {slide_id} already exists. Skipping ...')
            return
        self.log.debug(f'Processing {slide_id}')

        # Open required TIFs
        rgb_slide = openslide.open_slide(str(slide_obj_entry['rgb']))
        bg_slide = openslide.open_slide(str(slide_obj_entry['bg']))

        count = 0
        slide_coord_map = {'x': [], 'y': [], 'slide_id': []}

        for x, y, bg_tile in self.sliding_window(bg_slide):
            if self.tissue_threshold(bg_tile):
                slide_coord_map['x'].append(x)
                slide_coord_map['y'].append(y)
                slide_coord_map['slide_id'].append(slide_id)
                count += 1

        rgb_slide.close()
        bg_slide.close()

        df = pd.DataFrame(slide_coord_map)
        self.log.debug(f'Extracted {len(df)} patches.')
        self.log.debug(f'Saving to: "{str(output_fn)}"')

        # Save as pickle & compress
        df.to_pickle(output_fn, compression='gzip')

    def iterate_inputs(self, file_pattern='*.tif') -> Generator[dict, None, None]:
        """Iterates over RGB files and yields dictionaries containing paths"""
        for slide_rgb in self.rgb_dir.glob(file_pattern):
            slide_filename = slide_rgb.name
            slide_label = self.label_dir / slide_filename
            slide_bg = self.bg_dir / slide_filename

            if not (slide_rgb.exists() and slide_bg.exists()):
                self.log.debug('Skipping: rgb and bg already exist')
                continue
            yield {'rgb': slide_rgb, 'label': slide_label, 'bg': slide_bg}

    def sliding_window(self, slide: OSlide) -> Generator[Image, None, None]:
        """Reads reagions of an OpenSlide and yields PIL.Image"""
        region_size = (self.tile_size, self.tile_size)
        max_x, max_y = slide.level_dimensions[self.level]
        for cur_y in range(0, max_y, self.step_size):
            for cur_x in range(0, max_x, self.step_size):
                yield cur_x, \
                    cur_y, \
                    slide.read_region((cur_x, cur_y), self.level, region_size) \
                    .convert('L')

    def tissue_threshold(self, binary_im: Image):
        if self.min_tissue is None:
            self.min_tissue = -np.infty
        if self.max_tissue is None:
            self.max_tissue = np.infty
        np_binary_im = np.array(binary_im)
        return self.min_tissue < np.sum(np_binary_im.astype('bool')) / \
            np_binary_im.size < self.max_tissue

    def create_dataset(self) -> NoReturn:
        """Joins all DataFrames inside coord_maps folder into one.
        The merged DataFrame represents a dataset."""
        dfs = [(x.stem, pd.read_pickle(x))
               for x in self.coord_maps_dir.glob('*.gz')]

        # named_dfs = [df.assign(slide=slide_id) for slide_id, df in dfs]
        df = pd.concat([df for (_, df) in dfs])
        self.log.debug(f'{len(df)} tiles generated in total')
        ds_name = self.dataset_dir / f'{self.coord_maps_dir.stem}.gz'

        df.to_pickle(ds_name, compression='gzip')
        self.log.info(f'Dataset saved as: {ds_name}')

    def _init_dir_structure(self, data_base_dir: Path, ds_prefix: str) -> NoReturn:
        def encode_min_max(n: int) -> str:
            """Encodes float as string.
            e.g.: 0.75 -> "075"; 1.0 -> "1"
            """
            return '0' if n == 0 else str(round(n, 4)).replace('.', '').rstrip('0')

        self.data_base_dir = data_base_dir
        # self.dataset_name = dataset_name

        ds_name = f'L{self.level}-' \
                  f'T{self.tile_size}-' \
                  f'S{self.step_size}-' \
                  f'MIN{encode_min_max(self.min_tissue)}-' \
                  f'MAX{encode_min_max(self.max_tissue)}'

        if ds_prefix:
            ds_name = f'{ds_prefix}-{ds_name}'

        # Loads data from
        self.rgb_dir = data_base_dir / 'rgb'
        self.label_dir = data_base_dir / 'label'
        self.bg_dir = data_base_dir / 'bg'

        # Differentiate same datasets using a timestamp
        if (data_base_dir / 'coord_maps' / ds_name).exists():
            ds_name += '-' + str(round(time()))

        # Stores data to
        self.coord_maps_dir = mkdir(data_base_dir / 'coord_maps' / ds_name, exist_ok=True)
        self.dataset_dir = mkdir(data_base_dir / 'datasets', exist_ok=True)


if __name__ == '__main__':
    """
    Preprocesses raw data into coordinate maps
    and a dataset file for tissue segmentation.

    Example usage:
        $ python -m rationai.data.segment.create_map
                        --base_dir data/breast/
                        --level 1
                        --max_workers 20
    """
    parser = argparse.ArgumentParser()
    # Data collection
    parser.add_argument('-b', '--base_dir', type=Path, required=True, help='Data collection directory.')
    parser.add_argument('--ds_prefix', type=str, default='', help='Descriptive prefix of a newly created dataset.')
    parser.add_argument('--file_pattern', type=str, default='*', help='Input filenames pattern')

    # Metadata
    parser.add_argument('-l', '--level', type=int, required=True, help='Sampling level')
    parser.add_argument('-t', '--tile_size', type=int, default=512, help='Tile size - the side of a square in pixels.')
    parser.add_argument('-s', '--step_size', type=int, default=512, help='distance to move a sliding window at every step')
    parser.add_argument('--min_tissue', type=float, default=0.5, help='Minimum tissue proportion required for a tile to be included in a result set.')
    parser.add_argument('--max_tissue', type=float, default=1.0, help='Maximum tissue proportion required for a tile to be included in a result set.')

    # Performance
    parser.add_argument('--max_workers', type=int, default=1, help='Task parallelism (may consume lots of RAM).')

    args = parser.parse_args()
    CreateMap(
        data_base_dir=args.base_dir,
        ds_prefix=args.ds_prefix,
        file_pattern=args.file_pattern,
        level=args.level,
        tile_sile=args.tile_size,
        step_size=args.step_size,
        min_tissue=args.min_tissue,
        max_tissue=args.max_tissue,
        max_workers=args.max_workers
    ).run()
