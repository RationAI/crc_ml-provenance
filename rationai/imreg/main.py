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

# Type hints
from typing import NoReturn
from typing import Union
from rationai.imreg.magic_constants import PROCESSING_LEVEL
from rationai.imreg.sample_segmentation.segment_samples import get_samples_generator
from rationai.imreg.sample_segmentation.ignore_annotation import draw_multipolygon, draw_polygon
from rationai.imreg.our_method.piecewise_transformation import piecewise_transformation, transform_image_pwise
from rationai.imreg.our_method.utils.image_tools import apply_color_segmentation_quick_per_blocks, \
    transform_image_by_shapely_transform, get_stain_histomics
from rationai.imreg.our_method.utils.point_tools import transform_points
from rationai.imreg.our_method.utils.utils_generic import prepare_dir
from rationai.imreg.our_method.point_registration.point_registration_computation import compute_point_registration_transform
from rationai.imreg.our_method.registration_points.registration_points_computation import compute_registration_points
from rationai.imreg.others.util import compute_stain_matrix_hdab_histomics, compute_stain_matrix_he_histomics
from rationai.imreg.cytokeratin_mask.cytokeratin_mask_processing import create_cytokeratin_mask, fill_holes, create_he_mask,\
    mask_remove_he_background
from rationai.imreg.sample_segmentation.ignore_annotation import read_ignore_annotation_as_multipolygon


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

    def __init__(self,
                 he_mrxs_path: str,
                 hdab_mrxs_path: str,
                 output_dir: str,
                 verbose: bool = False):

        self.he_mrxs_path = he_mrxs_path
        self.hdab_mrxs_path = hdab_mrxs_path
        self.verbose = verbose

        self._init_dir_structure(output_dir)

    def _init_dir_structure(self, output_dir: str) -> NoReturn:
        """Saves original images to raw/ and masks to masks/"""

        # Create output directory path name
        he_stem = Path(self.he_mrxs_path).stem
        hdab_stem = Path(self.hdab_mrxs_path).stem

        self.he_dab_combined_name = f'{he_stem}--{hdab_stem}'
        output_path = os.path.join(output_dir, self.he_dab_combined_name)

        # Saves originals to
        self.rgb_dir = f'{output_path}/raw/'
        # Saves registered binary masks to
        self.label_dir = f'{output_path}/masks/'

        prepare_dir(self.rgb_dir)
        prepare_dir(self.label_dir)

    def process_slide(self,
                      he_openslide: openslide.OpenSlide,
                      ce_openslide: openslide.OpenSlide,
                      he_ignore_annotation: Union[MultiPolygon, None],
                      ce_ignore_annotation: Union[MultiPolygon, None]) -> NoReturn:

        t0 = time.time()

        # Create generator of WSI samples (cores)
        gen = get_samples_generator(he_openslide,
                                    ce_openslide,
                                    PROCESSING_LEVEL,
                                    he_ignore_annotation=he_ignore_annotation,
                                    ce_ignore_annotation=ce_ignore_annotation,
                                    select_samples=[])

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
        print(f'H&E cores saved to: {self.rgb_dir}')
        print(f'Registered masks saved to: {self.label_dir}')


    def process_pair(self,
                     he,
                     ce,
                     rgb_from_H_E,
                     rgb_from_H_DAB,
                     he_annot,
                     ce_annot,
                     i: int):
        if self.verbose:
            print(f'Processing pair {i}')
        h_he_stain = get_stain_histomics(he, rgb_from_H_E, 0)
        c_dab_stain = get_stain_histomics(ce, rgb_from_H_DAB, 1)

        he_mask = create_he_mask(h_he_stain)

        he_quick = apply_color_segmentation_quick_per_blocks(he)
        ce_quick = apply_color_segmentation_quick_per_blocks(ce)

        he_quick = get_stain_histomics(he_quick, rgb_from_H_E, 0)
        ce_quick = get_stain_histomics(ce_quick, rgb_from_H_DAB, 0)

        # Initial alignment
        fixed_points, moving_points, fixed_max, moving_max, _, _, _, _ = compute_registration_points(he_quick, ce_quick)
        transform = compute_point_registration_transform(fixed_points, moving_points)

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
        mask = img_as_float(create_cytokeratin_mask(c_dab_stain))
        mask_transformed = transform_image_by_shapely_transform(mask, transform)

        # Compute & use warp transformation (can be commented out to keep only rigid transformations)
        warp = piecewise_transformation(c_dab_stain.shape, transform_points(moving_max, transform), fixed_max, 6)
        mask_transformed = transform_image_pwise(img_as_bool(mask_transformed), warp)

        mask_transformed = fill_holes(mask_transformed, he_mask)

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
        mask_no_bg = mask_remove_he_background(img_as_ubyte(mask_transformed),
                                               img_as_ubyte(he))

        # Save images
        imsave(f'{str(self.rgb_dir)}/{self.he_dab_combined_name}_{str(i)}.png',
               img_as_ubyte(he))
        # imsave(output_directory + "originals/ce_" + str(i) + ".png", ce)
        imsave(f'{str(self.label_dir)}/{self.he_dab_combined_name}_{str(i)}.png',
               img_as_ubyte(mask_no_bg))

        if self.verbose:
            print(f'{self.he_dab_combined_name}_{i} processed')

    def run(self):
        """Runs the alignment method for a pair of WSIs."""
        # load whole slide images
        he_openslide = openslide.OpenSlide(self.he_mrxs_path)
        ce_openslide = openslide.OpenSlide(self.hdab_mrxs_path)

        he_annotation = None
        ce_annotation = None

        # If exists, read annotation of ignore regions on HE WSI
        he_annotation_path = self.he_mrxs_path.replace("mrxs", "xml")
        if os.path.exists(he_annotation_path):
            he_annotation = read_ignore_annotation_as_multipolygon(he_annotation_path, 0)
            # print("HE ignore annotation exists")

        # If exists, read annotation of ignore regions on HDAB WSI
        ce_annotation_path = self.hdab_mrxs_path.replace("mrxs", "xml")
        if os.path.exists(ce_annotation_path):
            ce_annotation = read_ignore_annotation_as_multipolygon(ce_annotation_path, 0)
            # print("HDAB ignore annotation exists")

        # Process the WSI pair
        self.process_slide(he_openslide, ce_openslide, he_annotation, ce_annotation)
