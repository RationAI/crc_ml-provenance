from __future__ import annotations

import numpy as np
import openslide
import os
import time
import tqdm
from multiprocessing import Pool
from pathlib import Path
from skimage import img_as_float, img_as_bool, img_as_ubyte
from skimage.io import imsave
from shapely.geometry import MultiPolygon
import logging
import json
import argparse
import pyvips

# Type hints
from typing import NoReturn
from typing import Union
from typing import Any

from rationai.data.imreg.muni.piecewise_transformation import piecewise_transformation
from rationai.data.imreg.muni.piecewise_transformation import transform_image_pwise

from rationai.data.imreg.muni.utils.image_tools import apply_color_segmentation_quick_per_blocks
from rationai.data.imreg.muni.utils.image_tools import transform_image_by_shapely_transform
from rationai.data.imreg.muni.utils.image_tools import get_stain_histomics

from rationai.data.imreg.muni.utils.point_tools import transform_points

from rationai.data.imreg.muni.utils.utils_generic import prepare_dir

from rationai.data.imreg.muni.point_registration.point_registration_computation import compute_point_registration_transform

from rationai.data.imreg.muni.registration_points.registration_points_computation import compute_registration_points

from rationai.data.imreg.muni.utils.utils_imreg import compute_stain_matrix_hdab_histomics
from rationai.data.imreg.muni.utils.utils_imreg import compute_stain_matrix_he_histomics

from rationai.data.imreg.muni.utils.cytokeratin_mask_processing import create_cytokeratin_mask
from rationai.data.imreg.muni.utils.cytokeratin_mask_processing import fill_holes
from rationai.data.imreg.muni.utils.cytokeratin_mask_processing import create_he_mask
from rationai.data.imreg.muni.utils.cytokeratin_mask_processing import mask_remove_he_background

from rationai.data.imreg.muni.sample_segmentation.ignore_annotation import read_ignore_annotation_as_multipolygon
from rationai.data.imreg.muni.sample_segmentation.ignore_annotation import draw_multipolygon
from rationai.data.imreg.muni.sample_segmentation.ignore_annotation import draw_polygon

from rationai.data.imreg.muni.sample_segmentation.segment_samples import get_samples_generator

from rationai.utils.config import ConfigProto

log = logging.getLogger('image-reg')
logging.basicConfig(level=logging.INFO,
                   format='[%(asctime)s][%(levelname).1s][%(process)d][%(filename)s][%(funcName)-25.25s] %(message)s',
                   datefmt='%d.%m.%Y %H:%M:%S')

class ImageRegistration:
    """Performs image registration of tissue cores
    found in a pair of H&E stained and DAB re-stained pair of mrxs slides.

    1. segments WSI to find pairs of tissue cores
    2. performs image registration for each pair of H&E and DAB cores
    3. generates binary mask (of epithelial regions) for each core

    Output:
       output_dir
            |
            |_ <slide_name>
            |   |
            |   |_ raw - (original H&E cores)
            |       |_ <slide_name>_0.png
            |       |_ <slide_name>_1.png
            |       |_ ...
            |
            |_ <slide_name>
                |
                |_ masks - (registered epithelial masks)
                    |_ <slide_name>_0.png
                    |_ <slide_name>_1.png
                    |_ ...
    """

    def __init__(self, config: ConfigProto, output_dir: Path):
        self.config = config
        self._init_dir_structure(output_dir)

    def _init_dir_structure(self, output_dir: str) -> NoReturn:
        """Saves original images to raw/ and masks to masks/"""

        # Saves final images
        self.out_he_dir = output_dir / self.config.group / 'he'
        self.out_ce_dir = output_dir / self.config.group / 'ce'

        prepare_dir(self.out_he_dir)
        prepare_dir(self.out_ce_dir)

    def process_slide(self,
                      he_openslide: openslide.OpenSlide,
                      ce_openslide: openslide.OpenSlide,
                      he_ignore_annotation: Union[MultiPolygon, None],
                      ce_ignore_annotation: Union[MultiPolygon, None]) -> NoReturn:

        t0 = time.time()

        # Create generator of WSI samples (cores)
        gen = get_samples_generator(he_openslide,
                                    ce_openslide,
                                    self.config.processing_level,
                                    he_ignore_annotation=he_ignore_annotation,
                                    ce_ignore_annotation=ce_ignore_annotation,
                                    select_samples=[],
                                    blur_radius=self.config.blur_radius,
                                    white_thresh=self.config.white_thresh,
                                    minimum_area=self.config.minimum_area,
                                    segmentation_level=self.config.segmentation_level,
                                    min_sample_area=self.config.min_sample_area)

        images = list(gen)
        images = [images[0]]  # for debug TODO

        he_images = [sample["he_sample"] for sample in images]
        ce_images = [sample["ce_sample"] for sample in images]

        he_annotations = [sample["he_annotation"] for sample in images]
        ce_annotations = [sample["ce_annotation"] for sample in images]

        he_smaller = [sample["he_smaller"] for sample in images]
        ce_smaller = [sample["ce_smaller"] for sample in images]

        # no arg distributes the work among all available CPUs
        print('Computing H&E and DAB stain matrices ... ', end='')
        with Pool() as pool:
            he_matrices = list(pool.map(compute_stain_matrix_he_histomics, he_smaller))
            ce_matrices = list(pool.map(compute_stain_matrix_hdab_histomics, ce_smaller))
            print('Done')

        print('Computing image registration')
        # Computationaly heavy task,
        # paralelism is already used within process_pair.
        for i in tqdm.tqdm(range(len(images))):
            self.process_pair(he_images[i],
                              ce_images[i],
                              he_matrices[i],
                              ce_matrices[i],
                              he_annotations[i],
                              ce_annotations[i],
                              i)

        print(f'Running time: { (time.time()-t0) // 60} minutes')
        print(f'H&E cores saved to: {self.out_he_dir}')
        print(f'Registered masks saved to: {self.out_ce_dir}')

    def process_pair(self,
                     he,
                     ce,
                     rgb_from_H_E,
                     rgb_from_H_DAB,
                     he_annot,
                     ce_annot,
                     i: int):
        if self.config.verbose:
            print(f'Processing pair {i}')
        h_he_stain = get_stain_histomics(he, rgb_from_H_E, 0)
        c_dab_stain = get_stain_histomics(ce, rgb_from_H_DAB, 1)

        he_mask = create_he_mask(h_he_stain)

        he_quick = apply_color_segmentation_quick_per_blocks(he,
            self.config.color_seg_kernel_size,
            self.config.color_seg_max_dist,
            self.config.color_seg_ratio
        )
        ce_quick = apply_color_segmentation_quick_per_blocks(ce,
            self.config.color_seg_kernel_size,
            self.config.color_seg_max_dist,
            self.config.color_seg_ratio
        )

        he_quick = get_stain_histomics(he_quick, rgb_from_H_E, 0)
        ce_quick = get_stain_histomics(ce_quick, rgb_from_H_DAB, 0)

        # Initial alignment
        fixed_points, moving_points, fixed_max, moving_max, _, _, _, _ = compute_registration_points(he_quick, ce_quick,
            self.config.hematoxylin_optimal_number_nuclei_for_piecewise,
            self.config.cytokeratin_optimal_number_nuclei_for_piecewise,
            self.config.nuclei_max_area, self.ce_nuclei_min_area,
            self.config.nuclei_seg_color_thr_min,
            self.config.nuclei_seg_color_thr_max,
            self.config.nuclei_seg_color_thr_steps
        )
        transform = compute_point_registration_transform(fixed_points, moving_points,
            self.config.number_of_angle_steps,
            self.config.angle_steps,
            self.config.number_of_steps_grid_search_exp,
            self.config.number_of_parallel_grids
        )

        # Create ignore masks from annotations
        he_ignore_mask = np.zeros(he.shape[:2])
        ce_ignore_mask = np.zeros(ce.shape[:2])

        if he_annot is not None:
            try:
                draw_polygon(he_annot, he_ignore_mask, 1, 0)
            except Exception:
                draw_multipolygon(he_annot, he_ignore_mask, 1, 0)
        if ce_annot is not None:
            try:
                draw_polygon(ce_annot, ce_ignore_mask, 1, 0)
            except Exception:
                draw_multipolygon(ce_annot, ce_ignore_mask, 1, 0)

        # Ignore annotated HDAB regions
        c_dab_stain[img_as_bool(ce_ignore_mask)] = 0

        # Create & transform cytokeratin mask
        mask = img_as_float(create_cytokeratin_mask(c_dab_stain, self.config.mask_min_area, self.config.holes_min_area))
        mask_transformed = transform_image_by_shapely_transform(mask, transform)

        # Compute & use warp transformation (can be commented out to keep only rigid transformations)
        warp = piecewise_transformation(c_dab_stain.shape, transform_points(moving_max, transform), fixed_max, 6)
        mask_transformed = transform_image_pwise(img_as_bool(mask_transformed), warp)

        mask_transformed = fill_holes(mask_transformed, he_mask, self.config.holes_max_area, self.config.holes_min_area)

        # Ignore annotated HE regions on cytokeratin mask
        mask_transformed[img_as_bool(he_ignore_mask)] = 0

        if ce_annot is not None:
            transformed_ce_ignore = transform_image_by_shapely_transform(ce_ignore_mask, transform)
            mask_transformed[img_as_bool(transformed_ce_ignore)] = 0
            he[img_as_bool(transformed_ce_ignore)] = 1

        # Ignore annotated HE regions on HE image
        he[img_as_bool(img_as_ubyte(he_ignore_mask))] = 1

        # Remove HE background from mask
        # TODO: deal with img_as_ubyte(he) conversion warning:
        # possible loss while float64 -> uint8
        mask_no_bg = mask_remove_he_background(cytokeratin_mask=img_as_ubyte(mask_transformed),
                                               he_image=img_as_ubyte(he),
                                               mask_min_area=self.config.mask_min_area)

        # TODO -- Safe as pyramidal TIFFs!!!
        # Save images
        he_out_fp = (self.out_he_dir / f'{self.slide_name}_{i}').with_suffix('.tif')
        self.save_as_tif(img_as_ubyte(he))

        ce_out_fp = (self.out_ce_dir / f'{self.slide_name}_{i}').with_suffix('.tif')
        self.save_as_tif(img_as_ubyte(ce))

        if self.config.verbose:
            print(f'{self.slide_name}_{i} processed')

    def save_as_tif(self, input_im, output_path):
        vips_im = pyvips.Image.new_from_memory(np.array(input_im.convert('RGBA')), input_im.size[0], input_im.size[1], 4, format='uchar')
        vips_im.tiffsave(str(output_path), bigtiff=True, compression=pyvips.enums.ForeignTiffCompression.DEFLATE, tile=True, tile_width=256, tile_height=256, pyramid=True)

    def get_hdab_slide(self, slide_name: str):
        hdab_slide_fp = (self.config.hdab_dir / slide_name).with_suffix(self.config.pattern)
        if hdab_slide_fp.exists():
            log.debug(f'[{slide_name}] HDAB slide found.')
            return hdab_slide_fp
        log.error(f'{slide_name} has no corresponding HDAB slide.')
        return None

    def run(self, he_slide_fp: Path):
        slide_name = he_slide_fp.stem
        hdab_slide_fp = self.get_hdab_slide(self, slide_name)

        """Runs the alignment method for a pair of WSIs."""
        # load whole slide images
        he_openslide = openslide.OpenSlide(he_slide_fp)
        ce_openslide = openslide.OpenSlide(hdab_slide_fp)

        he_annotation = None
        ce_annotation = None

        # If exists, read annotation of ignore regions on HE WSI
        he_annotation_path = he_slide_fp.with_suffix('.xml')
        if he_annotation_path.exists():
            he_annotation = read_ignore_annotation_as_multipolygon(he_annotation_path, 0)

        # If exists, read annotation of ignore regions on HDAB WSI
        ce_annotation_path = hdab_slide_fp.with_suffix('.xml')
        if ce_annotation_path.exists():
            ce_annotation = read_ignore_annotation_as_multipolygon(ce_annotation_path, 0)

        # Process the WSI pair
        self.process_slide(he_openslide, ce_openslide, he_annotation, ce_annotation)

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
        def __init__(self, config_fp, eid):
            self.config_fp = config_fp
            self.eid = eid
            self.verbose = False

            # Input Path Parameters
            self.he_dir = None
            self.hdab_dir = None
            self.pattern = None

            # Output Path Parameters
            self.output_path = None
            self.group = None

            # Segment Samples Parameters
            self.segmentation_level = None
            self.processing_level = None
            self.segmentation_closing_diam_he = None
            self.segmentation_closing_diam_ce = None

            # Slide Segmentation Parameters
            self.min_distance_from_edge = None
            self.min_distance_between_centroids = None
            self.max_distance_between_centroids = None
            self.blur_radius = None
            self.minimum_area = None
            self.white_thresh = None
            self.min_sample_area = None

            # Color Segmentation Parameters
            self.color_seg_kernel_size = None
            self.color_seg_max_dist = None
            self.color_seg_ratio = None

            # Segment Nuclei Parameters
            self.he_nuclei_max_area = None
            self.he_nuclei_min_area = None
            self.ce_nuclei_max_area = None
            self.ce_nuclei_min_area = None
            self.cytokeratin_optimal_number_nuclei = None
            self.hematoxylin_optimal_number_nuclei = None
            self.cytokeratin_optimal_number_nuclei_for_piecewise = None
            self.hematoxylin_optimal_number_nuclei_for_piecewise = None
            self.nuclei_seg_color_thr_min = None
            self.nuclei_seg_color_thr_max = None
            self.nuclei_seg_color_thr_steps = None

            # Fill Holes Parameters
            self.holes_min_area = None
            self.holes_max_Area = None
            self.holes_h_proportion = None
            self.mask_min_area = None

            # Hierarchical Grid Search Parameters
            self.number_of_parallel_grids = None
            self.number_of_steps_grid_search_exp = None
            self.top_step_size_grid_exp = None
            self.bot_step_size_grid_exp = None
            self.stopping_bad_suffix_length = None

            # Angle Search Parameters
            self.angle_step = None
            self.number_of_angle_steps = None

            # Local Search Parameters
            self.local_search_number_of_steps = None
            self.local_search_step_size = None

            # Holding changed values
            self._default_config = {}

            # Iterable State
            self._groups = None
            self._cur_group_configs = []

        def __iter__(self) -> ImageRegistration.Config:
            """Populates the config parameters with default values.

            Returns:
                SlideConverter.Config: SlideConverter.Config with `_global` values.
            """
            log.info('Populating default options.')
            with open(self.config_fp, 'r') as json_r:
                config = json.load(json_r)['image-registration']

            # Set config to default state
            self.__set_options(config.pop('_global'))
            self._default_config = {}

            # Prepare iterator variable
            self._groups = config
            return self

        def __next__(self) -> ImageRegistration.Config:
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
            self.he_dir = Path(self.slide_dir)
            if self.hdab_dir:
                self.hdab_dir = Path(self.hdab_dir)

def main(args):
    for cfg in ImageRegistration.Config(args.config_fp):
        img_reg = ImageRegistration(cfg, args.output_dir)
        for slide_fp in cfg.slide_dir.glob(cfg.pattern):
            img_reg.run(slide_fp)

if __name__=='__main__':
    description = 'Image Registration Algorithm'
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    parser.add_argument('--output_dir', type=Path, required=True, help='Path to output directory.')
    args = parser.parse_args()
    main(args)
