# Standard Imports
from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional
from typing import Tuple
from typing import List
from typing import Iterator
from typing import Any
from pathlib import Path
from datetime import datetime
import argparse
import logging
import shutil
import json
import copy

# Third-party Imports
import numpy as np
import warnings
import tables
from nptyping import NDArray
import pandas as pd
import openslide as oslide
from pandas.core.frame import DataFrame
from PIL import Image
from PIL import ImageDraw
from skimage import color
from skimage import filters
from skimage import morphology
from openslide import OpenSlide

# Local Imports
from rationai.utils.utils import read_polygons
from rationai.utils.utils import open_pil_image
from rationai.utils.config import ConfigProto
from rationai.utils.provenance import SummaryWriter

# Allows to load large images
Image.MAX_IMAGE_PIXELS = None

# Suppress tables names warning
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

log = logging.getLogger('slide-converter')
logging.basicConfig(level=logging.INFO,
                   format='[%(asctime)s][%(levelname).1s][%(process)d][%(filename)s][%(funcName)-25.25s] %(message)s',
                   datefmt='%d.%m.%Y %H:%M:%S')

sw_log = SummaryWriter.getLogger('provenance')

@dataclass
class ROITile:
    coord_x: int
    coord_y: int
    annot_coverage: float

class SlideConverter:
    """Worker Object for tile extraction from WSI.

       Class (static) variable `dataset_h5` is a workaround for multiprocessing to work.
       HDFStore file handler does not support multiple write access. Thus a single file
       handler needs to be passed. This filehandler cannot be pickled and therefore cannot
       be stored as an instance variable or passed as an argument to the `__call__` function.
    """
    def __init__(self, config: ConfigProto):
        self.config = config
        self.center_filter_np = self._get_center_filter()

    def __call__(self, slide_fp: Path) -> Tuple[str, pd.DataFrame, dict]:
        """Converts slide into a coordinate map of ROI Tiles.

        Args:
            slide_fp (Path): Path to WSI file.

        Returns:
            Tuple[str, pd.DataFrame, dict]: Returns a tuple of table key,
                coordinate map dataframe and metadat dictionary.
        """
        self.slide_name = slide_fp.stem

        annot_fp = self._get_annotations()
        oslide_wsi = self._open_slide(slide_fp)

        is_mode_valid = self._validate_mode(annot_fp)
        is_wsi_levels_valid = self._validate_wsi_levels(oslide_wsi)

        if not (is_mode_valid and is_wsi_levels_valid):
            return str(), pd.DataFrame(), dict()

        bg_mask_img = self._get_bg_mask(oslide_wsi, annot_fp)
        annot_mask_img = self._get_annot_mask(oslide_wsi, annot_fp)

        coord_map_df = self._tile_wsi_to_coord_map(oslide_wsi, bg_mask_img, annot_mask_img)
        table_key = self._get_table_key()
        metadata = self._get_table_metadata(slide_fp, annot_fp)

        oslide_wsi.close()
        return table_key, coord_map_df, metadata

    def _get_center_filter(self) -> Image.Image:
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

    def _get_annotations(self) -> Optional[Path]:
        """Builds a path to annotation file using slide name and supplied annotation dir path.

        Returns:
            Optional[Path]: Path to annotation file if it exists; otherwise None.
        """
        if self.config.negative_mode:
            return None

        annot_fp = (self.config.label_dir / self.slide_name).with_suffix('.xml').resolve()
        if annot_fp.exists():
            log.debug(f'[{self.slide_name}] Annotation XML found.')
            return annot_fp

        log.warning(f'[{self.slide_name}] Annotation XML not found.')
        if not self.config.strict_mode:
            self.config.negative_mode = True
            log.warning(f'Setting negative flag to {self.config.negative_mode}.')
        return None

    def _open_slide(self, slide_fp: Path) -> OpenSlide:
        """Opens WSI slide and returns handler.

        Args:
            slide_fp (Path): Path to WSI slide.

        Returns:
            OpenSlide: Handler to opened WSI slide.
        """
        logging.info(f'[{self.slide_name}] Opening slide: {str(slide_fp.resolve())}')
        return oslide.open_slide(str(slide_fp.resolve()))

    def _validate_mode(self, annot_fp: Path) -> bool:
        """Checks requirements for a chosen slide conversion mode.

        Args:
            annot_fp (Path): Path to annotation file.

        Returns:
            bool: True if requirements are met; otherwise False.
        """
        if annot_fp is None and self.config.strict_mode:
            return False
        return True

    def _validate_wsi_levels(self, oslide_wsi: OpenSlide) -> bool:
        """Checks if WSI contains enough levels for slide successful slide conversion.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.

        Returns:
            bool: True if requirements are met; otherwise False.
        """
        max_level = max(self.config.sample_level, self.config.bg_level)
        if oslide_wsi.level_count < (max_level + 1):
            log.error(f'[{self.slide_name}] WSI does not contain {max_level + 1} levels.')
            return False
        return True

    def _get_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Retrieves binary background mask.

        Mask is retrieved from disk if already present and force parameter is not set.
        Otherwise, the mask is drawn using image processing techniques on a WSI.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary background mask filtering background and highlighting tissue.
        """
        bg_mask_fp = self.config.output_dir / f'masks/bg/bg_final/{self.slide_name}.PNG'
        if bg_mask_fp.exists() and not self.config.force:
            bg_mask_img = open_pil_image(bg_mask_fp)
            if bg_mask_img is not None:
                return bg_mask_img

        bg_mask_img = self._create_bg_mask(oslide_wsi, annot_fp)
        self._save_mask(bg_mask_img, bg_mask_fp)
        return bg_mask_img

    def _get_annot_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Optional[Image.Image]:
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
        if self.config.negative_mode:
            return None

        annot_mask_fp = self.config.output_dir / f'masks/annotations/{self.slide_name}.PNG'
        if annot_mask_fp.exists() and not self.config.force:
            annot_mask_img = open_pil_image(annot_mask_fp)
            if annot_mask_img is not None:
                return annot_mask_img

        annot_mask_img = self._create_annot_mask(oslide_wsi, annot_fp)
        self._save_mask(annot_mask_img, annot_mask_fp)
        return annot_mask_img

    def _create_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
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
        log.info(f'[{self.slide_name}] Generating new background mask.')
        init_bg_mask_img = self._get_init_bg_mask(oslide_wsi)

        annot_bg_mask_img = self._get_annot_bg_mask(oslide_wsi, annot_fp)

        return self._combine_bg_masks(init_bg_mask_img, annot_bg_mask_img)

    def _get_init_bg_mask(self, oslide_wsi: OpenSlide) -> Image.Image:
        """Retrieves initial background mask created using image processing techniques.

        Mask is retrieved from disk if already present and force parameter is not set.
        Otherwise, the mask is drawn using image processing techniques.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.

        Returns:
            Image.Image: Binary background mask.
        """
        init_bg_mask_fp = self.config.output_dir / f'masks/bg/bg_init/{self.slide_name}.PNG'
        if init_bg_mask_fp.exists() and not self.config.force:
            init_bg_mask_img = open_pil_image(init_bg_mask_fp)
            if init_bg_mask_img is not None:
                return init_bg_mask_img

        init_bg_mask_img = self._create_init_bg_mask(oslide_wsi)
        self._save_mask(init_bg_mask_img, init_bg_mask_fp)
        return init_bg_mask_img

    def _create_init_bg_mask(self, oslide_wsi: OpenSlide) -> Image.Image:
        """Draws binary background mask using image processing techniques.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.

        Returns:
            Image.Image: Binary background mask.
        """
        log.info(f'[{self.slide_name}] Generating new initial background mask.')
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

    def _get_annot_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
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
        if self.config.negative_mode:
            return None
        annot_bg_mask_fp = self.config.output_dir / f'masks/bg/bg_annot/{self.slide_name}.PNG'
        if annot_bg_mask_fp.exists() and not self.config.force:
            annot_bg_mask_img = open_pil_image(annot_bg_mask_fp)
            if annot_bg_mask_img is not None:
                return annot_bg_mask_img

        annot_bg_mask_img = self._create_annot_bg_mask(oslide_wsi, annot_fp)
        self._save_mask(annot_bg_mask_img, annot_bg_mask_fp)
        return annot_bg_mask_img

    def _create_annot_bg_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Draws binary background mask using supplied annotation file.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary background mask.
        """
        log.info(f'[{self.slide_name}] Generating new annotation background mask.')
        annot_bg_mask_size = oslide_wsi.level_dimensions[self.config.bg_level]
        annot_bg_scale_factor = int(oslide_wsi.level_downsamples[self.config.bg_level])
        canvas_color = 'BLACK' if self.config.strict_mode else 'WHITE'
        return self._draw_annotation_mask(annot_fp, annot_bg_mask_size, annot_bg_scale_factor,
            include_keywords=self.config.include_keywords,
            exclude_keywords=self.config.exclude_keywords,
            canvas_color=canvas_color)

    def _draw_annotation_mask(self, annot_fp: Path, size: Tuple[int, int], scale_factor: int,
                               include_keywords: List[str], exclude_keywords: List[str],
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
                                              explicitly overruled by annotation file
                                    'BLACK' - area should be considered as negative unless
                                              explicitly overruled by annotation file

        Returns:
            Image.Image: Binary mask.
        """
        annot_mask_img, annot_mask_draw = self._prepare_empty_canvas(size, canvas_color)
        incl_polygons = read_polygons(annot_fp, scale_factor=scale_factor,
                                      keywords=include_keywords)
        log.debug(f'[{self.slide_name}] Include polygons ({include_keywords}): {incl_polygons}')
        self._draw_polygons_on_mask(incl_polygons, annot_mask_draw, polygon_color='WHITE')

        excl_polygons = read_polygons(annot_fp, scale_factor=scale_factor,
                                      keywords=exclude_keywords)
        log.debug(f'[{self.slide_name}] Exclude polygons ({exclude_keywords}): {excl_polygons}')
        self._draw_polygons_on_mask(excl_polygons, annot_mask_draw, polygon_color='BLACK')

        return annot_mask_img

    def _prepare_empty_canvas(self, size: Tuple[int, int],
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

    def _draw_polygons_on_mask(self, polygons: List[List[float]],
                                canvas_draw: ImageDraw.ImageDraw,
                                polygon_color: str) -> None:
        """Draws polygons on a canvas based on provided annotation file.

        Args:
            polygons (List[List[float]]): List of polygons extracted from annotation file.
            canvas_draw (ImageDraw.ImageDraw): ImageDraw reference to canvas.
        """
        for polygon in polygons:
            if len(polygon) < 3:
                log.warning(f'[{self.slide_name}] Polygon {polygon} skipped because it contains less than 3 vertices.')
                continue
            canvas_draw.polygon(xy=polygon, outline=(polygon_color), fill=(polygon_color))

    def _combine_bg_masks(self, init_bg_mask_img: Image, annot_bg_mask_img: Image) -> Image.Image:
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

        if not self.config.negative_mode:
            combined_bg_mask = combined_bg_mask & np.array(annot_bg_mask_img)

        return Image.fromarray(combined_bg_mask.astype(np.uint8) * 255, mode='L')

    def _create_annot_mask(self, oslide_wsi: OpenSlide, annot_fp: Path) -> Image.Image:
        """Draws binary annotation mask using supplied annotation file.

        Args:
            oslide_wsi (OpenSlide): Handler to WSI.
            annot_fp (Path): Path to annotation file.

        Returns:
            Image.Image: Binary annotation mask.
        """
        log.info(f'[{self.slide_name}] Generating annotation mask.')
        annot_bg_mask_size = oslide_wsi.level_dimensions[self.config.sample_level]
        annot_bg_scale_factor = int(oslide_wsi.level_downsamples[self.config.sample_level])
        canvas_color = 'BLACK'
        return self._draw_annotation_mask(annot_fp, annot_bg_mask_size, annot_bg_scale_factor,
            include_keywords=self.config.include_keywords,
            exclude_keywords=[],
            canvas_color=canvas_color)

    def _save_mask(self, img: Image, output_fp: Path) -> None:
        """Saves binary mask image on disk.

        Args:
            img (Image): Binary mask.
            output_fp (Path): Output filepath.
        """
        # TODO: Resolve inconsistency between output dir vs output path
        if not output_fp.parent.exists():
            output_fp.parent.mkdir(parents=True)
        img.save(str(output_fp), format='PNG')

    def _tile_wsi_to_coord_map(self, oslide_wsi: OpenSlide, bg_mask_img: Image,
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
            'coord_x': [],          # (int)  x-coordinate of a top-left pixel of the tile
            'coord_y': [],          # (int)  y-coordinate of a top-left pixel of the tile
            'annot_coverage': [],   # (float) annotation overlap ratio
            'is_cancer': [],         # (bool) cancer present in the center area of the tile
            'slide_name': []        # (str)  slide identifier (filename)
        }
        log.info(f'[{self.slide_name}] Initiating slide conversion.')
        for roi_tile in self._roi_cutter(oslide_wsi, bg_mask_img, annot_mask_img):
            coord_map['coord_x'].append(roi_tile.coord_x)
            coord_map['coord_y'].append(roi_tile.coord_y)
            coord_map['annot_coverage'].append(roi_tile.annot_coverage)
            coord_map['is_cancer'].append(roi_tile.annot_coverage > 0)
            coord_map['slide_name'].append(self.slide_name)
        log.info(f'[{self.slide_name}] Slide conversion complete. Extracted {len(coord_map["is_cancer"])} tiles.')

        return pd.DataFrame.from_dict(coord_map)

    def _roi_cutter(self, oslide_wsi: OpenSlide, bg_mask_img: Image,
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

        for (coord_x, coord_y) in self._tile_cutter(wsi_height, wsi_width):
            bg_tile_img = self._crop_mask_to_tile(bg_mask_img, coord_x, coord_y, effective_scale_factor)

            if not self._is_bg_contain_tissue(bg_tile_img):
                continue

            annot_tile_img = self._crop_mask_to_tile(annot_mask_img, coord_x, coord_y, 1)
            label = self._determine_label(annot_tile_img)

            yield ROITile(coord_x * sampling_scale_factor,
                  coord_y * sampling_scale_factor,
                  label)

    def _tile_cutter(self, wsi_height: int, wsi_width: int) -> Iterator[Tuple[int, int]]:
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

    def _crop_mask_to_tile(self, mask_img: Optional[Image.Image],
                            coord_x: int, coord_y: int,
                            scale_factor: int) -> Optional[Image.Image]:
        """Crops mask to a tile specified by coordinates scaled to appropriate resolution.

        Args:
            mask_img (Optional[Image.Image]): Image to be cropped.
            coord_x (int): Coordinate along x-axis.
            coord_y (int): Coordinate along y-axis.
            scale_factor (int): Used for scaling coordinates.

        Returns:
            Optional[Image.Image]: Cropped tiled or None
        """
        if mask_img is None:
            return None
        return mask_img.crop((int(coord_x // scale_factor),
                              int(coord_y // scale_factor),
                              int((coord_x + self.config.tile_size) // scale_factor),
                              int((coord_y + self.config.tile_size) // scale_factor)))

    def _is_bg_contain_tissue(self, tile_img: Image) -> bool:
        """Checks if tissue ratio in a tile falls within acceptable range.

        Args:
            tile_img (Image): Extracted tile, where non-zero element is considered a tissue.

        Returns:
            bool: True if tissue ratio falls within acceptable range; otherwise False.
        """
        tile_np = np.array(tile_img)
        tissue_coverage = self._calculate_tissue_coverage(tile_np, tile_np.size)
        return self.config.min_tissue <= tissue_coverage <= self.config.max_tissue

    def _calculate_tissue_coverage(self, tile_np: NDArray, size: int) -> float:
        """Calculates ratio of non-zero elements in a tile.

        Args:
            tile_np (NDArray): Extracted tile.
            size (int): length of a single side of a square area

        Returns:
            float: Ratio of non-zero elements.
        """
        tissue_count = np.count_nonzero(tile_np)
        return tissue_count / size

    def _determine_label(self, annot_tile_img: Image) -> float:
        """Calculates ratio of annotated (non-zero) elements in

        Args:
            annot_tile_img (Image): Binary annotation tile.

        Returns:
            float: Ratio of annotated (non-zero) elements w.r.t. the center
                   area of the tile.
        """
        if self.config.negative_mode:
            return 0.0

        annot_tile_np = np.array(annot_tile_img)
        center_annot_coverage = self._calculate_tissue_coverage(
            annot_tile_np & self.center_filter_np,
            self.config.center_size * self.config.center_size
        )

        return center_annot_coverage

    def _get_table_key(self) -> str:
        return f'{self.config.group}/{self.slide_name}'

    def _get_table_metadata(self, slide_fp: Path, annot_fp: Path) -> dict:
        """Saves the following metadata with the table:
                - tile_size      size of the extracted tiles
                - center_size    size of the labelled center area
                - slide_fp       WSI filepath
                - annot_fp       XML Annotation filepath
                - sample_level   resolution level at which tiles were sampled

                - tissue type   Shortened tissue type ID
                - patient_id    Pseudo-anonymized patient ID
                - case_id       Sequential scan ID for each patient
                - year          Year when scan was made
                - is_cancer     1 if slide contains cancer, otherwise 0

                Name convention for slides:
                    (tissue_type)-(year)_(patient_id)-(case_id)-(cancer).mrxs

                Example:
                    P-2019_1477-13-1.mrxs

        Args:
            slide_fp (Path): WSI filepath.
            annot_fp (Path): XML Annotation filepath
        """
        metadata = dict()
        metadata['slide_fp'] = str(slide_fp)
        metadata['annot_fp'] = str(annot_fp)
        metadata['tile_size'] = self.config.tile_size
        metadata['center_size'] = self.config.center_size
        metadata['sample_level'] = self.config.sample_level

        tissue_type, year_patient_id, case_id, is_cancer = self.slide_name.split('-')
        year, patient_id = year_patient_id.split('_')

        metadata['tissue_type'] = tissue_type
        metadata['patient_id'] = patient_id
        metadata['is_cancer'] = is_cancer
        metadata['case_id'] = case_id
        metadata['year'] = year

        return metadata

    class Config(ConfigProto):
        """Iterable config for create map.

        The supplied config consists of two parts:

            - `_global` group is a mandatory key specifying default values. Parameters
            that are either same for every input or change only for some inputs
            should go in here.

            - One or more named groups. Each group custom group contains a list
            defining input files and parameters specific to these files. The
            value of these parameters will override the value of parameter
            defined in `_global` group.

        The config is an iterable object. At every iteration the SlideConverter.Config
        first defaults back to `_global` values before being overriden by the
        input specific parameters.
        """
        def __init__(self, config_fp):
            self.config_fp = config_fp

            # Input Path Parameters
            self.slide_dir = None
            self.label_dir = None
            self.pattern = None

            # Output Path Parameters
            self.output_dir = None
            self.group = None

            # Tile Parameters
            self.tile_size = None
            self.step_size = None
            self.center_size = None

            # Resolution Parameters
            self.sample_level = None
            self.bg_level = None

            # Filtering Parameters
            self.include_keywords = None
            self.exclude_keywords = None
            self.min_tissue = None
            self.max_tissue = None
            self.disk_size = None

            # Tiling Modes
            self.negative_mode = False
            self.strict_mode = False
            self.force = False

            # Paralellization Parameters
            self.max_workers = None

            # Holding changed values
            self._default_config = {}

            # Iterable State
            self._groups = None
            self._cur_group_configs = []

        def __iter__(self) -> SlideConverter.Config:
            """Populates the config parameters with default values.

            Returns:
                SlideConverter.Config: SlideConverter.Config with `_global` values.
            """
            log.info('Populating default options.')
            with open(self.config_fp, 'r') as json_r:
                config = json.load(json_r)['slide-converter']

            # Set config to default state
            self.__set_options(config.pop('_global'))
            self._default_config = {}

            # Prepare iterator variable
            self._groups = config
            return self

        def __next__(self) -> SlideConverter.Config:
            """First resets back default values before overriding the input specific
            parameters.

            Raises:
                StopIteration: No more input directories left to be processed

            Returns:
                SlideConverter.Config: Fully populated SlideConverter.Config ready to be processed.
            """
            if not (self._groups or self._cur_group_configs):
                raise StopIteration
            # For each input dir we only want to override
            # attributes explicitely configured in JSON file.
            self.__reset_to_default()
            if not self._cur_group_configs:
                self.__get_next_group()
            self.__set_options(self._cur_group_configs.pop())
            self.__validate_options()

            log.info(f'Now processing ({self.group}):{self.slide_dir}')
            return self

        def __set_options(self, partial_config: dict) -> None:
            """Iterates over the variable names and values pairs a setting
            the corresponding instance variables to these values.

            Args:
                config (dict): Partial configuration specifying variables
                            and values to be overriden
            """
            for k,v in partial_config.items():
                self.__set_option(k, v)

        def __set_option(self, k: str, v: Any) -> None:
            """Sets instance variable `k` with values `v`.

            Args:
                k (str): name of the instance variable
                v (Any): value to be set
            """
            if hasattr(self, k):
                self._default_config[k] = getattr(self, k)
                setattr(self, k, v)
            else:
                log.warning(f'Attribute {k} does not exist.')

        def __reset_to_default(self) -> None:
            """Reverts the overriden values back to the default values.
            """
            # Reset to global state
            while self._default_config:
                k,v = self._default_config.popitem()
                setattr(self, k, v)

        def __get_next_group(self) -> None:
            """Retrieves the next named group from the JSON config.
            """
            group, configs = self._groups.popitem()
            self.group = group
            self._cur_group_configs = configs

        def __validate_options(self) -> None:
            """Converts string paths to Path objects.
            """
            # Path attributes
            self.slide_dir = Path(self.slide_dir)
            if self.label_dir:
                self.label_dir = Path(self.label_dir)
            self.output_dir = Path(self.output_dir)

def main(args):
    dataset_h5 = None

    # Spawn worker for each slide; maximum `max_workers` simultaneous workers.
    for cfg in SlideConverter.Config(args.config_fp):
        if not cfg.output_dir.exists():
            cfg.output_dir.mkdir(parents=True)

        if dataset_h5 is None:
            dataset_h5 = pd.HDFStore((cfg.output_dir / cfg.output_dir.name).with_suffix('.h5'), 'w')
            # Copy configuration file
            shutil.copy2(args.config_fp, cfg.output_dir / args.config_fp.name)
            sw_log.set('config_file',  value=str((cfg.output_dir / args.config_fp.name).resolve()))
            sw_log.set('dataset_file', value=str(((cfg.output_dir / cfg.output_dir.name).with_suffix(".h5")).resolve()))

        log.info(f'Spawning {cfg.max_workers} workers.')
        with Pool(cfg.max_workers) as p:
            for table_key, table, metadata in p.imap(SlideConverter(copy.deepcopy(cfg)), list(cfg.slide_dir.glob(cfg.pattern))):
                if not table.empty:
                    dataset_h5.append(table_key, table)
                    dataset_h5.get_storer(table_key).attrs.metadata = metadata

    sw_log.to_json((cfg.output_dir / 'prov_preprocess.log').resolve())
    dataset_h5.close()


if __name__ == '__main__':

    description = """
    Slide conversion tool creates coordinate maps (table of coordinates) from the input slides.

    Example:
        python3 create_map.py path/to/config.json

             config_fp - Path to config file
    """

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    args = parser.parse_args()
    main(args)
