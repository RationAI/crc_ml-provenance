# Standard Imports
from collections import namedtuple
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional
from typing import Tuple
from typing import List
from typing import Iterator
from pathlib import Path
import argparse
import logging
import copy
import os

# Third-party Imports
import numpy as np
from nptyping import NDArray
import pandas as pd
from pandas.core.frame import DataFrame
from PIL import Image
from PIL import ImageDraw
from skimage import color
from skimage import filters
from skimage import morphology
from openslide import OpenSlide

# Local Imports
from rationai.data.utils import mkdir
from rationai.data.utils import read_polygons
from rationai.data.utils import open_pil_image
from rationai.data.classify.create_map_config import CreateMapConfig

# Allows to load large images
Image.MAX_IMAGE_PIXELS = None

log = logging.getLogger('slide-converter')

@dataclass
class ROITile:
    coord_x: int
    coord_y: int
    annot_coverage: float
    center_annot_coverage: float

class SlideConverter:

    def __init__(self, config: CreateMapConfig):
        self.config = config
        self.center_filter_np = self.__get_center_filter()
        self.__prepare_dir_structure()
        self.config.toJSON(self.config.output_path / 'config.json')

    def __prepare_dir_structure(self):
        # Base output dir
        self.config.output_path.mkdir(mode=0o770, parents=True, exist_ok=True)

        # Create mask directories
        masks_dir = self.config.output_path / 'masks'
        masks_dir.mkdir(mode=0o770, parents=True, exist_ok=True)
        (masks_dir / 'bg' / 'bg_init').mkdir(mode=0o770, parents=True, exist_ok=True)
        (masks_dir / 'bg' / 'bg_final').mkdir(mode=0o770, parents=True, exist_ok=True)
        (masks_dir / 'bg' / 'bg_annot').mkdir(mode=0o770, parents=True, exist_ok=True)
        (masks_dir / 'annotations').mkdir(mode=0o770, parents=True, exist_ok=True)

        # Create coord maps directories
        coord_maps_dir = self.config.output_path / 'coord_maps'
        coord_maps_dir.mkdir(mode=0o770, parents=True, exist_ok=True)
        log.info(f'[{os.getpid()}] Output location: {coord_maps_dir}')

    def __call__(self, slide_fp: Path) -> None:
        """Converts slide into a coordinate map of ROI Tiles.

        Args:
            slide_fp (Path): Path to WSI file.
        """
        self.slide_name = slide_fp.stem

        annot_fp = self.__get_annotations()
        oslide_wsi = self.__open_slide(slide_fp)

        is_mode_valid = self.__validate_mode()
        is_wsi_levels_valid = self.__validate_wsi_levels(oslide_wsi)

        if not (is_mode_valid and is_wsi_levels_valid):
            return None

        bg_mask_img = self.__get_bg_mask(oslide_wsi, annot_fp)
        annot_mask_img = self.__get_annot_mask(oslide_wsi, annot_fp)

        coord_map_df = self.__tile_wsi_to_coord_map(oslide_wsi, bg_mask_img, annot_mask_img)

        self.__save_coord_map(coord_map_df)
        oslide_wsi.close()

    def __get_center_filter(self) -> Image.Image:
        """Creates a binary mask for a tile, with a non-zero center square in the middle.

        Returns:
            Image.Image: Binary tile mask.
        """
        offset_size = int((self.config.tile_size - self.config.center_size) // 2)
        center_filter = Image.new('L', (self.config.tile_size, self.config.tile_size), 'BLACK')
        filter_draw = ImageDraw.Draw(center_filter)
        filter_draw.rectangle(
            [(offset_size, offset_size),
             (self.config.tile_size - offset_size, self.config.tile_size - offset_size)], 'WHITE')
        return center_filter

    def __get_annotations(self) -> Optional[Path]:
        """Builds a path to annotation file using slide name and supplied annotation dir path.

        Returns:
            Optional[Path]: Path to annotation file if it exists; otherwise None.
        """
        if self.config.negative:
            return None

        annot_fp = (self.config.label_dir / self.slide_name).with_suffix('.xml')
        if annot_fp.exists():
            log.info(f'[{os.getpid()}] Annotation XML found.')
            return annot_fp

        if not self.config.strict:
            self.config.negative = True
        log.info(f'[{os.getpid()}] Annotation XML not found.')
        return None

    def __open_slide(self, slide_fp: Path) -> OpenSlide:
        """Opens WSI slide and returns handler.

        Args:
            slide_fp (Path): Path to WSI slide.

        Returns:
            OpenSlide: Handler to opened WSI slide.
        """
        return OpenSlide(str(slide_fp.resolve()))

    def __validate_mode(self, annot_fp: Path) -> bool:
        """Checks requirements for a chosen slide conversion mode.

        Args:
            annot_fp (Path): Path to annotation file.

        Returns:
            bool: True if requirements are met; otherwise False.
        """
        if annot_fp is None and self.config.strict:
            return False
        return True

    def __validate_wsi_levels(self, oslide_wsi: OpenSlide) -> bool:
        """Checks if WSI contains enough levels for slide successful slide conversion.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.

        Returns:
            bool: True if requirements are met; otherwise False.
        """
        max_level = max(self.config.sample_level, self.config.bg_level)
        if oslide_wsi.level_count < (max_level + 1):
            log.error(f'[{os.getpid()}] WSI {self.slide_name} does not contain {max_level + 1} levels.')
            return False
        return True

    def __get_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Retrieves binary background mask.

        Mask is retrieved from disk if already present and force parameter is not set.
        Otherwise, the mask is drawn using image processing techniques on a WSI.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary background mask filtering background and highlighting tissue.
        """
        bg_mask_fp = self.config.output_path / f'masks/bg/bg_final/{self.slide_name}.PNG'
        if bg_mask_fp.exists() and not self.config.force:
            bg_mask_img = self.__load_image_from_file(bg_mask_fp)
            if bg_mask_img is not None: return bg_mask_img

        bg_mask_img = self.__create_bg_mask(oslide_wsi, annot_fp)
        self.__save_mask(bg_mask_img, bg_mask_fp)
        return bg_mask_img

    def __get_annot_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Optional[Image.Image]:
        """Retrieves binary annotation mask.

        Mask is retrieved from disk if already present and force parameter is not set.
        Otherwise, the mask is drawn using supplied annotation file.
        No mask is returned if slide conversion mode is set to 'Negative'.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Optional[Image.Image]: Binary annotation mask highlighting regions of interest;
                             None if conversion mode set to 'Negative'
        """
        if self.config.negative:
            return None

        annot_mask_fp = self.config.output_path / f'masks/annotations/{self.slide_name}.PNG'
        if annot_mask_fp.exists() and not self.config.force:
            annot_mask_img = self.__load_image_from_file(annot_mask_fp)
            if annot_mask_img is not None: return annot_mask_img

        annot_mask_img = self.__create_annot_mask(oslide_wsi, annot_fp)
        self.__save_mask(annot_mask_img, annot_mask_fp)
        return annot_mask_img

    def __load_image_from_file(self, image_fp: Path) -> Image.Image:
        """Loads image from disk.

        Args:
            image_fp (Path): Path to image file.

        Returns:
            Image.Image: Retrieved image.
        """
        log.info(f'[{os.getpid()}] Opening existing image: {image_fp}.')
        return open_pil_image(image_fp)

    def __create_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Creates binary background mask.

        Background mask is created by combining two masks:
             1) Mask obtained using image processing techniques applied to WSI
             2) Mask obtained using annotation file (if exists).

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary background mask.
        """
        log.info(f'[{os.getpid()}] Generating new background mask.')
        init_bg_mask_img = self.__get_init_bg_mask(oslide_wsi)

        annot_bg_mask_img = self.__get_annot_bg_mask(oslide_wsi, annot_fp)

        return self.__combine_bg_masks(init_bg_mask_img, annot_bg_mask_img)

    def __get_init_bg_mask(self, oslide_wsi: OpenSlide) -> Image.Image:
        """Retrieves initial background mask created using image processing techniques.

        Mask is retrieved from disk if already present and force parameter is not set.
        Otherwise, the mask is drawn using image processing techniques.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.

        Returns:
            Image.Image: Binary background mask.
        """
        init_bg_mask_fp = self.config.output_path / f'masks/bg/bg_init/{self.slide_name}.PNG'
        if init_bg_mask_fp.exists() and not self.config.force:
            init_bg_mask_img = self.__load_image_from_file(init_bg_mask_fp)
            if init_bg_mask_img is not None: return init_bg_mask_img

        init_bg_mask_img = self.__create_init_bg_mask(oslide_wsi)
        self.__save_mask(init_bg_mask_img, init_bg_mask_fp)
        return init_bg_mask_img

    def __create_init_bg_mask(self, oslide_wsi: OpenSlide) -> Image.Image:
        """Draws binary background mask using image processing techniques.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.

        Returns:
            Image.Image: Binary background mask.
        """
        log.info(f'[{os.getpid()}] Generating new initial background mask.')
        wsi_img = oslide_wsi.read_region(
            location=(0, 0),
            level=self.config.bg_level,
            size=oslide_wsi.level_dimensions[self.config.bg_level]).convert('RGB')
        slide_hsv = color.rgb2hsv(np.array(wsi_img))
        saturation = slide_hsv[:, :, 1]
        threshold = filters.threshold_otsu(saturation)
        high_saturation = (saturation > threshold)
        disk_object = morphology.disk(self.config.disk_size)
        mask = morphology.closing(high_saturation, disk_object)
        mask = morphology.opening(mask, disk_object)
        return Image.fromarray(mask)

    def __get_annot_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Retrieves binary background mask created using annotation file.

        Mask is retrieved from disk if already present and force parameter is not set.
        Otherwise, the mask is drawn using supplied annotation file.
        No mask is returned if slide conversion mode is set to 'Negative'.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary background mask.
        """
        if self.config.negative:
            return None
        annot_bg_mask_fp = self.config.output_path / f'masks/bg/bg_annot/{self.slide_name}.PNG'
        if annot_bg_mask_fp.exists() and not self.config.force:
            annot_bg_mask_img = self.__load_image_from_file(annot_bg_mask_fp)
            if annot_bg_mask_img is not None: return annot_bg_mask_img

        annot_bg_mask_img = self.__create_annot_bg_mask(oslide_wsi, annot_fp)
        self.__save_mask(annot_bg_mask_img, annot_bg_mask_fp)
        return annot_bg_mask_img

    def __create_annot_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Draws binary background mask using supplied annotation file.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary background mask.
        """
        log.info(f'[{os.getpid()}] Generating new annotation background mask.')
        annot_bg_mask_size = oslide_wsi.level_dimensions[self.config.bg_level]
        annot_bg_scale_factor = int(oslide_wsi.level_downsamples[self.config.bg_level])
        canvas_color = 'BLACK' if self.config.strict else 'WHITE'
        return self.__draw_annotation_mask(annot_fp, annot_bg_mask_size, annot_bg_scale_factor, \
            include_keywords=self.config.include_keywords, \
            exclude_keywords=self.config.exclude_keywords, \
            canvas_color=canvas_color)

    def __draw_annotation_mask(self, annot_fp: Path, size: Tuple[int, int], scale_factor: int, \
                               include_keywords: List[str], exclude_keywords: List[str], \
                               canvas_color: str) -> Image.Image:
        """Draws binary mask using supplied annotation file.

        Args:
            annot_fp (Path): Path to annotation file.
            size (Tuple[int, int]): Size of the mask to be drawn.
            scale_factor (int): Scaling factor for coordinates in annotation file.
            include_keywords (List[str]): Keywords corresponding to entries in annotation file
                                          that should be drawn as positive (white) areas.
            exclude_keywords (List[str]): Keywords corresponding to entries in annotation file
                                          that should be drawn as negative (black) areas.
            canvas_color (str): Default canvas color. Describes implicit behaviour:
                                    'WHITE' - area should be considered as positive unless
                                              explicitely overruled by annotation file
                                    'BLACK' - area should be considered as negative unless
                                              explicitely overruled by annotation file

        Returns:
            Image.Image: Binary mask.
        """
        annot_mask_img, annot_mask_draw = self.__prepare_empty_canvas(size, canvas_color)
        incl_polygons = read_polygons(annot_fp, scale_factor=scale_factor, \
                                      keywords=include_keywords)
        self.__draw_polygons_on_mask(incl_polygons, annot_mask_draw, color='WHITE')

        excl_polygons = read_polygons(annot_fp, scale_factor=scale_factor, \
                                      keywords=exclude_keywords)
        self.__draw_polygons_on_mask(excl_polygons, annot_mask_draw, color='BLACK')

        return annot_mask_img

    def __prepare_empty_canvas(self, size: Tuple[int, int], \
                               bg_color: str) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        """Prepares an empty canvas with default colour.

        Args:
            size (Tuple[int, int]): Size of a canvas.
            bg_color (str): Default colour of a canvas.

        Returns:
            Tuple[Image.Image, ImageDraw.ImageDraw]: Empty canvas and handler enabling drawing
                                                     on the canvas.
        """
        canvas = Image.new('L', size=size, color=bg_color)
        draw = ImageDraw.Draw(canvas)
        return canvas, draw

    def __combine_bg_masks(self, init_bg_mask_img: Image, annot_bg_mask_img: Image) -> Image.Image:
        """Combines two binary masks using binary AND operation.

        If slide conversion mode is set to 'Negative', initial background mask is returned.

        Args:
            init_bg_mask_img (Image): Binary background mask obtained using image processing
                                      techniques.
            annot_bg_mask_img (Image): Binary background mask obtained using supplied annotation
                                       file.

        Returns:
            Image.Image: Combined binary background mask.
        """
        combined_bg_mask = np.array(init_bg_mask_img)

        if not self.config.negative:
            combined_bg_mask = combined_bg_mask & np.array(annot_bg_mask_img)

        return Image.fromarray(combined_bg_mask.astype(np.uint8) * 255, mode='L')

    def __create_annot_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Draws binary annotation mask using supplied annotation file.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary annotation mask.
        """
        log.info(f'[{os.getpid()}] Generating annotation mask.')
        annot_bg_mask_size = oslide_wsi.level_dimensions[self.config.sample_level]
        annot_bg_scale_factor = int(oslide_wsi.level_downsamples[self.config.sample_level])
        canvas_color = 'BLACK'
        return self.__draw_annotation_mask(annot_fp, annot_bg_mask_size, annot_bg_scale_factor, \
            include_keywords=self.config.include_keywords, \
            exclude_keywords=[], \
            canvas_color=canvas_color)

    def __save_mask(self, img: Image, output_fp: Path) -> None:
        """Saves binary mask image on disk.

        Args:
            img (Image): Binary mask.
            output_fp (Path): Output filepath.
        """
        img.save(str(output_fp), format='PNG')

    def __tile_wsi_to_coord_map(self, oslide_wsi: OpenSlide, bg_mask_img: Image, \
                                annot_mask_img: Image) -> DataFrame:
        """Builds a coordinate map dataframe using extracted ROI tiles.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            bg_mask_img (Image): Binary background mask.
            annot_mask_img (Image): Binary annotation mask.

        Returns:
            DataFrame: Coordinate map of ROI tiles.
        """
        coord_map = {
            'coord_x': [],
            'coord_y': [],
            'tumor_tile': [],
            'center_tumor_tile': [],
            'slide_name': []
        }
        log.info(f'[{os.getpid()}] Converting slide: {self.slide_name}')
        for roi_tile in self.__roi_cutter(oslide_wsi, bg_mask_img, annot_mask_img):
            coord_map['coord_x'].append(roi_tile.coord_x)
            coord_map['coord_y'].append(roi_tile.coord_y)
            coord_map['tumor_tile'].append(roi_tile.annot_coverage)
            coord_map['center_tumor_tile'].append(roi_tile.center_annot_coverage)
            coord_map['slide_name'].append(self.slide_name)

        return pd.DataFrame.from_dict(coord_map)

    def __roi_cutter(self, oslide_wsi: OpenSlide, bg_mask_img: Image, \
                     annot_mask_img: Image) -> Iterator[ROITile]:
        """Filters extracted tiles based on tissue coverage.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            bg_mask_img (Image): Binary background mask.
            annot_mask_img (Image): Binary annotation mask.

        Yields:
            Iterator[ROITile]: Iterator over extracted ROI tiles meeting all filtering requirements.
                               ROI Tile contains the following information:
                                    - coordinates of a tile (top-left pixel)
                                    - ratio of annotated pixels w.r.t. the whole tile
                                    - ratio of annotated pixels w.r.t. the center square of the tile
        """
        # Scale Factors
        bg_scale_factor = int(oslide_wsi.level_downsamples[self.config.bg_level])
        sampling_scale_factor = int(oslide_wsi.level_downsamples[self.config.sample_level])
        effective_scale_factor = bg_scale_factor // sampling_scale_factor

        # Dimensions
        wsi_width, wsi_height = oslide_wsi.level_dimensions[self.config.sample_level]

        for (coord_x, coord_y) in self.__tile_cutter(wsi_height, wsi_width):
            bg_tile_img = self.__crop_mask_to_tile(bg_mask_img, coord_x, coord_y, effective_scale_factor)
            if not self.__is_tile_contain_tissue(bg_tile_img):
                continue

            annot_tile_img = self.__crop_mask_to_tile(annot_mask_img, coord_x, coord_y, 1)
            annot_coverage, center_annot_coverage = self.__determine_label(annot_tile_img)

            yield ROITile(coord_x * sampling_scale_factor, \
                  coord_y * sampling_scale_factor, \
                  annot_coverage, \
                  center_annot_coverage)

    def __tile_cutter(self, wsi_height: int, wsi_width: int) -> Iterator[Tuple[int, int]]:
        """Iterates over tile coordinates of a WSI.

        Args:
            wsi_height (int): WSI height.
            wsi_width (int): WSI width.

        Yields:
            Iterator[Tuple[int, int]]: Iterator over tile coordinates.
        """
        for coord_y in range(0, wsi_height, self.config.step_size):
            for coord_x in range(0, wsi_width, self.config.step_size):
                yield coord_x, coord_y

    def __crop_mask_to_tile(self, mask_img, coord_x, coord_y, scale_factor) -> Image.Image:
        """Crops mask to a tile specified by coordinates scaled to appropriate resolution.

        Args:
            mask_img ([type]): Mask to be cropped.
            coord_x ([type]): Coordinate along x-axis.
            coord_y ([type]): Coordinate along y-axis.
            scale_factor ([type]): Scaling factor for coordinates.

        Returns:
            Image.Image: Extracted tile.
        """
        return mask_img.crop((int(coord_x // scale_factor),
                              int(coord_y // scale_factor),
                              int((coord_x + self.config.tile_size) // scale_factor),
                              int((coord_y + self.config.tile_size) // scale_factor)))

    def __is_tile_contain_tissue(self, tile_img: Image) -> bool:
        """Checks if tissue ratio in a tile falls within acceptable range.

        Args:
            tile_img (Image): Extracted tile, where non-zero element is considered a tissue.

        Returns:
            bool: True if tissue ratio falls within acceptable range; otherwise False.
        """
        tile_np = np.array(tile_img)
        tissue_coverage = self.__calculate_tissue_coverage(tile_np)
        return self.config.min_tissue <= tissue_coverage <= self.config.max_tissue

    def __calculate_tissue_coverage(self, tile_np: NDArray) -> float:
        """Calculates ratio of non-zero elements in a tile.

        Args:
            tile_np (NDArray): Extracted tile.

        Returns:
            float: Ratio of non-zero elements.
        """
        tissue_count = np.count_nonzero(tile_np)
        all_count = np.size(tile_np)
        return tissue_count / all_count

    def __determine_label(self, annot_tile_img: Image) -> Tuple[float, float]:
        """Calculates ratio of annotated (non-zero) elements in

        Args:
            annot_tile_img (Image): Binary annotation tile.

        Returns:
            Tuple[float, float]: Tuple of ratios of annotated (non-zero) elements w.r.t. the entire
            tile and w.r.t. the center area of the tile.
        """
        if self.config.negative:
            return 0.0, 0.0

        annot_tile_np = np.array(annot_tile_img)
        annot_coverage = self.__calculate_tissue_coverage(annot_tile_np)
        center_annot_coverage = self.__calculate_tissue_coverage(annot_tile_np & self.center_filter_np)

        return annot_coverage, center_annot_coverage

    def __save_coord_map(self, coord_map_df: DataFrame) -> None:
        """Saves coordinate map dataframe on a disk.

        Args:
            coord_map_df (DataFrame): Coordinate map dataframe.
        """
        coord_map_fp = self.config.output_path / f'coord_maps/{self.slide_name}.gz'
        coord_map_df.to_pickle(coord_map_fp, compression='gzip')
        log.info(f'[{os.getpid()}] Coord map with {len(coord_map_df)} ROI tiles saved at: {coord_map_fp}.')

def main(args):
    cfg = CreateMapConfig(args.config_fp)
    log.info(f'Creating {cfg.max_workers} workers.')
    with Pool(cfg.max_workers) as p:
        p.map(SlideConverter(copy.copy(cfg)), cfg.slide_dir.glob(cfg.pattern))
    return True


if __name__ == '__main__':

    description = """
    Slide conversion tool creates coordinate maps (table of coordinates) from the input slides.

    Example:
        python3 create_map.py path/to/config.json

             config_fp - Path to config file
    """

    parser = argparse.ArgumentParser(description=description, \
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    args = parser.parse_args()
    main(args)
