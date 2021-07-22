import argparse
import logging
from enum import Enum

# import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool
from openslide import OpenSlide
from openslide import OpenSlideError
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from skimage import color
from skimage import filters
from skimage import morphology
from nptyping import NDArray
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from rationai.data.utils import (
    mkdir,
    read_polygons,
    open_pil_image
)

# Allows to load large images
Image.MAX_IMAGE_PIXELS = None

log = logging.getLogger('slide-converter')


class XMLResult(Enum):
    OK = 0
    SWITCH_TO_NEGATIVE = 1
    SKIP_SLIDE = 2


# Prostate XML annotation keywords
# TODO: make configurable
INCLUDE_ANNOT_KEYWORDS = ['Carcinoma', 'metastases']
EXCLUDE_ANNOT_KEYWORDS = ['Exclude', 'Another pathology']


class SlideConverter:

    def __init__(self, args):

        # Required variables
        self.data_base_dir = args.base_dir  # data group name
        self.slide_dir = self.data_base_dir / args.slide_dir  # rgb/ (.mrxs)
        self.label_dir = self.data_base_dir / args.label_dir  # label/ (.xml)

        self.ds_prefix = args.ds_prefix

        # Sampling variables -> subdir: L1-T512-S128-C256-MIN05-MAX10
        self.level = args.level
        self.tile_size = args.tile_size
        self.step_size = args.step_size
        self.center_size = args.center_size
        self.min_tissue = args.min_tissue
        self.max_tissue = args.max_tissue

        # Mode variables
        self.negative_mode = args.negative
        self.strict_mode = args.strict
        self.force = args.force

        # Utility variables
        self.bg_level = args.bg_level
        self.percent = 0
        self.MORPHOLOGY_DISK_SIZE = 10

        self.center_filter = self._get_center_filter()

        # Output directory for coord_maps
        self.ds_dir = self._create_ds_dir()

    def __call__(self, slide_fn: Path):
        negative_mode = self.negative_mode

        slide_fn = slide_fn.resolve()
        slide_name = slide_fn.stem

        if (self.ds_dir / f'{slide_name}.gz').exists() and not self.force:
            log.debug(f'{slide_name}.gz already exists. Skipping.')
            return

        log.info('Processing: ' + slide_name)

        # Get annotation XML path
        result, annot_xml_filepath = self._get_xml_filepath(slide_name)
        if result is XMLResult.SKIP_SLIDE:
            return None
        elif result is XMLResult.SWITCH_TO_NEGATIVE:
            negative_mode = True

        # Open WSI slide
        try:
            slide = OpenSlide(str(slide_fn))
        except OpenSlideError as e:
            log.warning(f'Could not open slide {slide_fn}: {e}')
            return

        scale_factor, bg_scale_factor, effective_scale_factor = \
            self.get_slide_scale_factors(slide)

        # Skip if only single level
        if slide.level_count < 2:
            log.warning('Not enough levels. Skipping...')
            return None

        # Get background and cancer masks
        bg_mask = self.get_bg_mask(slide, slide_name, bg_scale_factor, annot_xml_filepath, negative_mode=negative_mode)
        cancer_mask = self.get_cancer_mask(slide, slide_name, scale_factor, annot_xml_filepath) \
            if not negative_mode else None

        log.info(f'Processing WSI tiles (tile_size={self.tile_size})')
        slide_width, slide_height = slide.level_dimensions[self.level]

        offset_map = {
            'coord_y': [],
            'coord_x': [],
            'tumor_tile': [],
            'center_tumor_tile': [],
            'slide_name': []
        }

        for coord_y in range(0, slide_height, self.step_size):
            for coord_x in range(0, slide_width, self.step_size):

                # Progress meter
                # NOTE: what is this?
                # percent = self.calculate_progress(coord_y, coord_x, slide_width, slide_height)

                # Retrieve a tile from background mask
                tile_mask = bg_mask.crop((int(coord_x // effective_scale_factor),
                                          int(coord_y // effective_scale_factor),
                                          int((coord_x + self.tile_size) // effective_scale_factor),
                                          int((coord_y + self.tile_size) // effective_scale_factor)))

                # Skip to next tile if condition for tissue tile is not met
                tissue_coverage = self.tissue_percent(
                    np.array(tile_mask),
                    int(self.tile_size // effective_scale_factor))

                if not self.min_tissue <= tissue_coverage <= self.max_tissue:
                    continue

                tile_label, center_label = self.determine_label(cancer_mask,
                                                                coord_x,
                                                                coord_y,
                                                                negative_mode)

                offset_map['coord_y'].append(coord_y * scale_factor)
                offset_map['coord_x'].append(coord_x * scale_factor)
                offset_map['tumor_tile'].append(tile_label)
                offset_map['center_tumor_tile'].append(center_label)
                offset_map['slide_name'].append(slide_name)

        offset_df = pd.DataFrame.from_dict(offset_map)

        if len(offset_df) > 0:
            log.debug(f'Saving {slide_name} (total={len(offset_df)}, '
                      f'tile_size={self.tile_size}, negative={negative_mode}, strict={self.strict_mode})')
            offset_df.to_pickle(self.ds_dir / f'{slide_name}.gz', compression='gzip')
        else:
            log.warning(f'{slide_name} produced an empty coordinate maps.')

    def _create_ds_dir(self) -> Path:
        """Creates directory for coord_maps.
        Name contains dataset prefix and sampling parameters"""
        def encode_min_max(n: int) -> str:
            """Encodes float as string.
            e.g.: 0.75 -> "075"; 1.0 -> "1"
            """
            return '0' if n == 0 else str(round(n, 4)).replace('.', '').rstrip('0')

        dataset_name = f'L{self.level}-' \
                       f'T{self.tile_size}-' \
                       f'S{self.step_size}-' \
                       f'C{self.center_size}-' \
                       f'MIN{encode_min_max(self.min_tissue)}-' \
                       f'MAX{encode_min_max(self.max_tissue)}'

        if self.ds_prefix:
            dataset_name = f'{self.ds_prefix}-{dataset_name}'

        ds_dir = self.data_base_dir / 'coord_maps' / dataset_name

        # if ds_dir.exists():
        #     raise ValueError(f'Dataset "{ds_dir}" already exists')
        mkdir(ds_dir)
        return ds_dir

    def _get_xml_filepath(self, slide_name: Union[str, Path]) -> Tuple[XMLResult, Optional[Path]]:
        # Case 1: Negative slide, no XML is required
        if self.negative_mode:
            return XMLResult.OK, None

        annot_xml_filepath = (self.label_dir / slide_name).with_suffix('.xml')

        # Case 2: XML exists
        if annot_xml_filepath.exists():
            return XMLResult.OK, annot_xml_filepath.resolve()

        # Case 3: XML does not exist and strict mode
        if self.strict_mode:
            log.warning(f'Annotation {annot_xml_filepath.name} does not exist and strict mode is on. Skipping slide...')
            return XMLResult.SKIP_SLIDE, None

        # Case 4: XML does not exist and not strict mode
        log.warning(f'Annotation {annot_xml_filepath.name} does not exist. Switching to negative mode...')
        return XMLResult.SWITCH_TO_NEGATIVE, None

    def _get_center_filter(self):
        """Creates a square mask in the centre of the tile.
        The size of the square is given by center_size attribute."""
        offset_size = int((self.tile_size - self.center_size) // 2)
        center_filter = Image.new('L', (self.tile_size, self.tile_size), 'BLACK')
        filter_draw = ImageDraw.Draw(center_filter)
        filter_draw.rectangle(
            [(offset_size, offset_size),
             (self.tile_size - offset_size, self.tile_size - offset_size)], 'WHITE')
        return center_filter

    def determine_label(self,
                        cancer_mask: Image,
                        coord_x: int,
                        coord_y: int,
                        negative_mode: bool) -> Tuple[float, bool]:
        # All tiles are negative for healthy patients
        if negative_mode:
            return 0.0, False

        # Retrieve a tile from cancer mask
        tile_mask = cancer_mask.crop((coord_x,
                                      coord_y,
                                      coord_x + self.tile_size,
                                      coord_y + self.tile_size))

        # DETERMINE TILE LABEL
        # percentage (to allow dynamic thresholding)
        tile_label = self.tissue_percent(np.array(tile_mask), self.tile_size)
        # boolean label: is True if at least one pixel in the center is annotated
        center_label = self.tissue_percent(
            np.array(tile_mask) & self.center_filter, 1) > 0

        return tile_label, center_label

    def calculate_progress(self, coord_y, coord_x, slide_width, slide_height):
        """
            Calculate progress as a ratio of processed slide area vs total slide area
        """
        total = slide_height * slide_width
        processed = (coord_y * slide_width + self.center_size * coord_x)
        percentage = int(processed / total * 100)

        if percentage > self.percent:
            pass
            # print('Progress: {:3}%\r'.format(percentage), end='')

        return percentage

    def get_slide_scale_factors(self, slide: OpenSlide) -> Tuple[float, float, float]:
        scale_factor = int(slide.level_downsamples[self.level])
        log.debug(f'Level {self.level} scaling: {scale_factor}')

        bg_scale_factor = int(slide.level_downsamples[self.bg_level])
        log.debug(f'BG Level {self.bg_level} scaling: {bg_scale_factor}')

        effective_scale_factor = bg_scale_factor // scale_factor
        log.debug(f'Effective scaling: {effective_scale_factor}')
        return scale_factor, bg_scale_factor, effective_scale_factor

    def tissue_percent(self, tile_mask: NDArray, tile_size: int) -> float:
        """
            Calculates the fraction of non-black area vs total area in a mask
        """
        ts_count = np.count_nonzero(tile_mask)
        bg_count = tile_size ** 2
        return ts_count / bg_count

    def create_cancer_annotation(self,
                                 annotation_fp: Union[str, Path],
                                 size: Tuple[int, int],
                                 scale_factor: float) -> Image:
        return self.get_annotation_mask(
            Path(annotation_fp),
            include_keywords=INCLUDE_ANNOT_KEYWORDS,
            exclude_keywords=[],
            size=size,
            scale_factor=scale_factor,
            bg_color='BLACK')

    def get_cancer_mask(self,
                        slide: OpenSlide,
                        slide_name: str,
                        scale_factor: float,
                        annot_xml_filepath: Union[str, Path]) -> Image:
        # Create mask dir if does not exist
        cancer_mask_dir = self.data_base_dir / f'masks/annotations-level{self.level}'
        cancer_mask_filepath = cancer_mask_dir / f'{slide_name}.png'

        if cancer_mask_filepath.exists():
            img = open_pil_image(cancer_mask_filepath)
            if img:
                return img

        mkdir(cancer_mask_dir, exist_ok=True)

        # Create new annotation mask
        cancer_mask = self.create_cancer_annotation(
            annot_xml_filepath,
            size=slide.level_dimensions[self.level],
            scale_factor=scale_factor)
        cancer_mask.save(str(cancer_mask_filepath), format='PNG')

        return cancer_mask

    def create_bg_mask(self, slide: OpenSlide, bg_level: int) -> Image:
        slide_img = slide.read_region(
            location=(0, 0),
            level=bg_level,
            size=slide.level_dimensions[bg_level]).convert('RGB')
        slide_hsv = color.rgb2hsv(np.array(slide_img))
        saturation = slide_hsv[:, :, 1]
        threshold = filters.threshold_otsu(saturation)
        high_saturation = (saturation > threshold)
        disk_object = morphology.disk(self.MORPHOLOGY_DISK_SIZE)
        mask = morphology.closing(high_saturation, disk_object)
        mask = morphology.opening(mask, disk_object)
        return Image.fromarray(mask)

    def create_bg_annotation(self,
                             annotation_fp: Union[str, Path],
                             size: Tuple[int, int],
                             scale_factor: float) -> Image:
        return self.get_annotation_mask(annotation_fp,
                                        INCLUDE_ANNOT_KEYWORDS,
                                        EXCLUDE_ANNOT_KEYWORDS,
                                        size=size,
                                        scale_factor=scale_factor,
                                        bg_color='BLACK' if self.strict_mode else 'WHITE')

    def get_bg_mask(self, slide, slide_name, bg_scale_factor, annot_xml_filepath, negative_mode):
        # Create mask dir if does not exist
        masks_dir = self.data_base_dir / f'masks/bg-level{self.bg_level}'
        mkdir(masks_dir, exist_ok=True)

        # COMBINED BACKGROUND MASK PROCESS #
        final_bg_dir = masks_dir / 'bg_final'
        final_bg_filepath = final_bg_dir / f'{slide_name}.png'
        if final_bg_filepath.exists() and not self.force:
            img = open_pil_image(final_bg_filepath)
            if img:
                return img

        mkdir(final_bg_dir, exist_ok=True)

        # BACKGROUND MASK PROCESS #
        log.debug(f'"{final_bg_filepath}" does not exists. Creating new mask.')
        bg_mask_dir = masks_dir / 'bg_init'
        bg_mask_filepath = bg_mask_dir / \
            f'{slide_name}-bg-level{self.bg_level}-hsv-otsu-disk{self.MORPHOLOGY_DISK_SIZE}-close-open.png'
        mkdir(bg_mask_dir, exist_ok=True)

        log.debug(f'bg_mask_filepath exits [{bg_mask_filepath.exists()}]: {bg_mask_filepath}')
        # Can be reused because it depends only on level parameter
        # which is present directly in the filename.
        if bg_mask_filepath.exists():
            log.debug('Init background mask already exists. Loading from disk.')
            bg_mask = Image.open(str(bg_mask_filepath))
        else:
            bg_mask = self.create_bg_mask(slide, self.bg_level)
            bg_mask.save(str(bg_mask_filepath), format='PNG')

        # ANNOTATION MASK PROCESS #
        if not negative_mode:
            annot_dir = masks_dir / 'bg_annot'
            annot_fp = annot_dir / f'{slide_name}.png'

            if annot_fp.exists() and not self.force:
                annot_mask = Image.open(str(annot_fp))
            else:
                mkdir(annot_dir)
                annot_mask = self.create_bg_annotation(
                    annot_xml_filepath,
                    size=slide.level_dimensions[self.bg_level],
                    scale_factor=bg_scale_factor)
                annot_mask.save(str(annot_fp), format='PNG')

        # Combine masks
        if negative_mode:
            combined_bg_mask = Image.fromarray(
                np.array(bg_mask).astype(np.uint8) * 255, mode='L')
        else:
            combined_bg_mask = Image.fromarray(
                (np.array(bg_mask) & np.array(annot_mask)).astype(np.uint8) * 255, mode='L')
        combined_bg_mask.save(str(final_bg_filepath), format='PNG')
        return combined_bg_mask

    def get_annotation_mask(self,
                            annot_filepath: Path,
                            include_keywords: List[str],
                            exclude_keywords: List[str],
                            size: Tuple[int, int],
                            scale_factor: float = 1,
                            bg_color: str = 'BLACK') -> Image:
        """
            Creates a binary mask for the cancer area (white) from annotation file
        """
        log.info('Creating annotation mask.')
        mask = Image.new('L', size=size, color=bg_color)
        draw = ImageDraw.Draw(mask)

        incl_polygons, excl_polygons = read_polygons(
            annot_filepath, scale_factor, include_keywords, exclude_keywords)

        for polygon in incl_polygons:
            if len(polygon) < 2:
                log.warning('Polygon skipped because it contains a single vertex.')
                continue
            draw.polygon(xy=polygon, outline=('WHITE'), fill=('WHITE'))

        for polygon in excl_polygons:
            if len(polygon) < 2:
                log.warning('Polygon skipped because it contains a single vertex.')
                continue
            draw.polygon(xy=polygon, outline=('BLACK'), fill=('BLACK'))

        return mask


def main(args):
    with Pool(args.max_workers) as p:
        p.map(SlideConverter(args),
              (args.base_dir / args.slide_dir).glob(args.pattern))
    return True


if __name__ == '__main__':

    description = """
    This script creates a coordination map for a WSI slide.

    [EXAMPLE USAGE]
    python3 -m rationai.data.classify.create_map \
        --base_dir data/Prostate \
        --slide_dir slides \
        --ds_prefix my_dataset \
        --level 1 \
        --tile_size 512 \
        --step_size 128 \
        --center_size 256 \
        --max_workers 10 \
        -f

    The coordination map is a pandas table exported and compressed to
    .gz file which consists of columns:
        coord_x             - x coordinate of a tile
        coord_y             - y coordinate of a tille
        center_tumor_tile   - whether there is cancer within the center square (bool)
        tumor_tile          - percentage instead of bool
        slide_name          - the stem of a source WSI filename

    Each row represents a single tile of the slide. Column and row defines
    the coordinates of the tile, while binary column label indicates whether
    the tile represents a tile containing a cancer tissue.

    The negative mode - should be used for healthy patient slides. These
                        slides are generally unannotatedand all tissue
                        areas are implicitely considered healthy. For this
                        reason, all XML annotations are ignored and only
                        background masks are generated and used in segmentation.

    The strict mode - should be used when only explicitely annotated tissue
                      areas should be present in the output.

    The force mode - causes coordinate maps and intermediate data to be generated again
                     instead of skipping existing processed coordinate maps
                     or reusing existing intermediate data.
                     Should be used when intermediate data already exists for a dataset
                     generated with a different set of processing parameters.
                     An exception is reusage of background masks of WSIs which
                     are not affected by this flag.
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required variables
    parser.add_argument('--base_dir', type=Path, required=True, help='Data collection directory')

    parser.add_argument('--slide_dir', type=Path, default=Path('rgb'),
                        help='Directory containing slides. (Can be relative to base_dir)')
    parser.add_argument('--label_dir', type=Path, default=Path('label'),
                        help='Directory containing annotations. (Can be relative to base_dir)')

    parser.add_argument('--ds_prefix', type=str, required=True,
                        help='Prefix of a newly created dataset.')

    parser.add_argument('-l', '--level', type=int, default=1, help='Sampling level')
    parser.add_argument('-t', '--tile_size', type=int, default=512, help='Tile size - the side of a square in pixels.')
    parser.add_argument('-s', '--step_size', type=int, default=512, help='Window step size (tile overlapping parameter).')
    parser.add_argument('-c', '--center_size', type=int, default=256,
                        help='Size of a tile "center" - a tile is positive if a square in its center contains annotated area.')
    parser.add_argument('--min_tissue', type=float, default=0.5,
                        help='Minimum tissue proportion required for a tile to be included '
                             'in a result set.')
    parser.add_argument('--max_tissue', type=float, default=1.0,
                        help='Maximum tissue proportion required for a tile to be included '
                             'in a result set.')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Task parallelism (may consume lots of RAM)')

    parser.add_argument('--bg_level', type=int, default=4,
                        help='Background level for tissue masks')

    # Mode variables
    parser.add_argument('-n', '--negative', action='store_true', required=False,
                        help='Folder only contains negative slides. '
                             'The algorithm will not require annotations to parse slides.')
    parser.add_argument('-S', '--strict', action='store_true', required=False,
                        help='Use only annotated areas. Do not infer negative tiles.')
    parser.add_argument('-f', '--force', action='store_true', required=False,
                        help='Causes existing files to be regenerated instead of reused.')

    # Process only files matching the given pattern
    parser.add_argument('-p', '--pattern', required=False, default='*.mrxs',
                        help="Limits processing to files matching the pattern. (default: '*.mrxs')")

    args = parser.parse_args()

    # Annotations are required only if negative mode is NOT set.
    if not args.negative and args.label_dir is None:
        raise ValueError('If --negative is not set, --annotations is required.')

    main(args)
