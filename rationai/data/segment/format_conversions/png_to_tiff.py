from __future__ import annotations

import argparse
import logging
import numpy as np
import pyvips
import tqdm

from multiprocessing import Pool
from pathlib import Path
from PIL import Image

# Image modules
from skimage import filters
from skimage import morphology
from skimage.color import rgb2hsv

# Type hints
from typing import Dict
from typing import Generator
from typing import NoReturn

from rationai.data.segment.format_conversions import ConverterBase
from rationai.data.utils import mkdir

# Allows to load large images
Image.MAX_IMAGE_PIXELS = None


# TODO: rework to make conversion of multiple dirs into single DS possible
class Png2Tiff(ConverterBase):
    """Converts preprocessed PNG images to TIFF format.
    Subclass tailored for cytokeratin image registration output structure conversion.

    source_dir                  - searches for patterns: '**/*.png'
        |_<slide_1>                                      '**/*.png'
        |       |_ raw
        |       |   |_ he_0.png
        |       |   |_ he_1.png
        |       |
        |       |_ masks
        |           |_ mask_0.png
        |           |_ mask_1.png
        |_<slide_2>


    output_base_dir
        |_ rgb
        |   |_ <slide_1>_<filename.stem>.tif
        |
        |_ label
        |   |_ <slide_1>_<filename.stem>.tif
        |
        |_ bg
            |_ <slide_1>_<filename.stem>.tif
    """
    def __init__(self,
                 source_dir: Path,       # looks for PNGs in ./<dir>/raw/ and masks/
                 output_base_dir: Path,  # saves TIFFs to rgb/ label/ bg/
                 tile_size: int,
                 max_workers: int = 1,
                 force_overwrite: bool = False,
                 verbose=0):

        super().__init__(source_dir=source_dir,
                         output_base_dir=output_base_dir,
                         max_workers=max_workers)

        self.tile_size = tile_size
        self.force_overwrite = force_overwrite
        self.verbose = verbose

        self.log = logging.getLogger('Png2Tiff')
        self._init_dir_structure(self.source_dir, self.output_base_dir)

    @classmethod
    def from_params(cls, params) -> Png2Tiff:
        # TODO: implement if StepExecutor wants to use this class
        raise NotImplementedError('from_params is not implemented')

    def run(self):
        self.log.info('Running PNG to TIFF conversion')

        with Pool(self.max_workers) as pool:
            gen = self.get_batch_generator()

            if self.verbose:
                # tqdm is not used in verbose mode
                pool.map(self.convert, gen)
            else:
                array = list(gen)
                # imap is used instad of map for easier tqdm utilization.
                # imap does lazy execution. Returned iterator has to be used.
                [_ for _ in tqdm.tqdm(pool.imap(self.convert, array),
                                      total=len(array))]

        self.log.info('PNG to TIFF conversion finished')

    def convert(self, input_dict: Dict[str, Path]) -> NoReturn:
        """Converts single batch of inputs to TIF"""
        rbg_path = input_dict['rgb_path']
        label_path = input_dict['label_path']

        input_image_name = self._get_output_filename(rbg_path)

        message = f'Processing: {input_image_name}'
        if self.verbose:
            self.log.info(message)
        else:
            self.log.debug(message)

        output_rgb_filepath = (self.out_rgb_dir / input_image_name).with_suffix('.tif')
        output_label_filepath = (self.out_label_dir / input_image_name).with_suffix('.tif')
        output_bg_filepath = (self.out_bg_dir / input_image_name).with_suffix('.tif')

        # Check if already exists
        if (not self.force_overwrite) and \
           (output_rgb_filepath.exists() and
                output_label_filepath.exists() and
                output_bg_filepath.exists()):
            self.log.debug(f'WARNING: {input_image_name} already converted and '
                           f' force_overwrite={self.force_overwrite}.')
            return

        # Create PIL.Image instances
        rgb_im = Image.open(rbg_path)
        label_im = Image.open(label_path)
        bg_im = self.create_bg_mask(rgb_im)

        # Save PIL.Image as .TIF files on disk
        self.save_image_as_tif(rgb_im, output_rgb_filepath)
        self.save_image_as_tif(bg_im, output_bg_filepath)
        self.save_image_as_tif(label_im, output_label_filepath)

    def create_bg_mask(self, img: Image) -> Image:
        """Generates new background mask"""
        im_hsv = rgb2hsv(np.array(img))
        im_sat = im_hsv[:, :, 1]
        threshold = filters.threshold_otsu(im_sat)
        im_sat_thresh = (im_sat > threshold)
        disk_object = morphology.disk(10)
        mask = morphology.closing(im_sat_thresh, disk_object)
        mask = morphology.opening(mask, disk_object)
        return Image.fromarray(mask)

    # Dir structure & file naming specific
    def get_batch_generator(self) -> Generator[Dict[str, Path], None, None]:
        """Generator yields a pair of paths: src_rgb, src_label.
        Specific for cytokeratin image registration output image naming scheme!!!"""
        for src_rgb in self.src_rgb_dir.glob('**/*.png'):
            # id number
            img_id = src_rgb.stem.strip().split('_')[-1]
            # name without id number
            core_name_no_id = src_rgb.stem.strip().split('_')[0]
            src_label = self.src_label_dir / f'{core_name_no_id}_{img_id}.png'
            if not src_label.exists():
                self.log.debug(f'WARNING: {src_rgb.name} has no mask.')
                continue
            yield {'rgb_path': src_rgb, 'label_path': src_label}

    # Dir structure & file naming specific
    def _get_output_filename(self, im_path: Path) -> Path:
        """Returns output file name for given source image.
        Example: <src_rgb_dir>/HE-KOS02--DAB-CK-KOS02/masks/mask_{int}.png
            src_filename:  HE-KOS02--DAB-CK-KOS02_0.png
            src_group_name: HE-KOS02--DAB-CK-KOS02
            """
        # src_filename = im_path.stem
        # src_group_name = im_path.parent.parent.stem
        # return Path(f'{src_group_name}_{src_filename}')
        return im_path.stem

    def save_image_as_tif(self, img: Image, output_path: Path) -> NoReturn:
        self.log.debug(f'Saving to {output_path}')
        vips_im = pyvips.Image.new_from_memory(data=np.array(img.convert('RGBA')),
                                               width=img.size[0],
                                               height=img.size[1],
                                               bands=4,
                                               format='uchar')
        vips_im.tiffsave(str(output_path),
                         bigtiff=True,
                         compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
                         tile=True,
                         tile_width=self.tile_size,
                         tile_height=self.tile_size,
                         pyramid=True)

    def _init_dir_structure(self, source_dir: Path, output_base_dir: Path) -> NoReturn:
        """Defines structure where data are read from and saved to"""
        # Looks for PNGs in subfolder of:
        self.src_rgb_dir = source_dir / 'raw'
        self.src_label_dir = source_dir / 'masks'

        # Saves TIFFs to:
        self.out_rgb_dir = mkdir(output_base_dir / 'rgb',   exist_ok=True)
        self.out_label_dir = mkdir(output_base_dir / 'label', exist_ok=True)
        self.out_bg_dir = mkdir(output_base_dir / 'bg',    exist_ok=True)


if __name__ == '__main__':
    """
    Example usage:
        $ python -m rationai.data.segment.format_conversions.png_to_tiff \
                        --input_dir /mnt/data/cytoseg/data/ \
                        --output_dir data/crc-breast\
                        --tile_size 256 \
                        --max_workers 10

    """

    parser = argparse.ArgumentParser()

    # Compulsory args
    parser.add_argument('--input_dir',   type=Path,  help='Base dir containing raw/ masks/ with PNG files')
    parser.add_argument('--output_dir',  type=Path,  help='Base dir where to write TIFFs (data group dir)')
    parser.add_argument('--tile_size',   type=int,   help='Size of tiles in pixels')

    # Optional args
    parser.add_argument('--max_workers',
                        type=int,
                        default=1,
                        help='Max number of multiprocessing workers')
    parser.add_argument('--force',
                        action='store_true',
                        help='Overwrites existing files if set')
    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='Verbosity level')

    args = parser.parse_args()

    converter = Png2Tiff(source_dir=args.input_dir,
                         output_base_dir=args.output_dir,
                         tile_size=args.tile_size,
                         max_workers=args.max_workers,
                         force_overwrite=args.force,
                         verbose=args.verbose)

    converter.run()
