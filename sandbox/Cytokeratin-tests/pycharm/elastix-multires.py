import os
import time

from registration_points.registration_points_computation import compute_registration_points

t0 = time.time()

import openslide
from skimage.io import imsave

from utils.image_tools import superimpose_mask_on_image
from sample_segmentation.segment_samples import get_samples_generator
from point_registration.point_registration_computation import compute_point_registration_transform

from cytokeratin_mask.hematoxylin_with_cytokeratin_mask import \
    register_hematoxylin_and_cytokeratin_mask_with_filled_holes
from magic_constants import MANUAL_CROP_X_BOUNDS
import pickle

SAVE_IMAGES = True
COMPUTE_POINTS = True
COMPUTE_TRANSFORM = True

output_directory = 'output_new/'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

he_openslide = openslide.OpenSlide("/mnt/data/scans/AI scans/Cytokeratin_mask_test/HE-HR2-16PgR-B.mrxs")
ce_openslide = openslide.OpenSlide("/mnt/data/scans/AI scans/Cytokeratin_mask_test/CK-DAB-H-HR2-16PgR-B.mrxs")

he_pre_crop = (MANUAL_CROP_X_BOUNDS["lower"],
               0,
               MANUAL_CROP_X_BOUNDS["upper"],
               he_openslide.dimensions[1])
ce_pre_crop = (MANUAL_CROP_X_BOUNDS["lower"],
               0,
               MANUAL_CROP_X_BOUNDS["upper"],
               ce_openslide.dimensions[1])

gen = get_samples_generator(he_openslide, ce_openslide, 2, he_pre_crop, ce_pre_crop)

if (SAVE_IMAGES):
    if not os.path.exists(output_directory + "segments/"):
        os.makedirs(output_directory + "segments/")
    for i, sample in enumerate(gen):
        he = sample["he_sample"]
        ce = sample["ce_sample"]
        print("Saving images")
        imsave(output_directory + "segments/h" + str(i) + ".png", sample["he_sample"])
        imsave(output_directory + "segments/c" + str(i) + ".png", sample["ce_sample"])
        print("Images saved")

