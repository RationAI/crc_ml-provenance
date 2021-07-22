import os
import time

from registration_points.registration_points_computation import compute_registration_points

t0 = time.time()

import openslide
from skimage.io import imsave, imread

from utils.image_tools import superimpose_mask_on_image, apply_color_segmentation_quick_per_blocks
from sample_segmentation.segment_samples import get_samples_generator
from point_registration.point_registration_computation import compute_point_registration_transform

from cytokeratin_mask.hematoxylin_with_cytokeratin_mask import \
    register_hematoxylin_and_cytokeratin_mask_with_filled_holes
from magic_constants import MANUAL_CROP_X_BOUNDS, PROCESSING_LEVEL

import pickle

SAVE_IMAGES = True
COMPUTE_COLOR_SEGMENTATION = True
COMPUTE_POINTS = True
COMPUTE_TRANSFORM = True
SELECT_SAMPLES = []

output_directory = 'output_13092019-1/'

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

gen = get_samples_generator(he_openslide, ce_openslide, PROCESSING_LEVEL, he_pre_crop, ce_pre_crop, SELECT_SAMPLES)

if (SAVE_IMAGES):
    if not os.path.exists(output_directory + "segments/"):
        os.makedirs(output_directory + "segments/")
    for i, sample in enumerate(gen):
        he = sample["he_sample"]
        ce = sample["ce_sample"]
        imsave(output_directory + "segments/h" + str(i) + ".png", sample["he_sample"])
        imsave(output_directory + "segments/c" + str(i) + ".png", sample["ce_sample"])

    gen = get_samples_generator(he_openslide, ce_openslide, PROCESSING_LEVEL, he_pre_crop, ce_pre_crop, SELECT_SAMPLES)

# %%
for i, sample in enumerate(gen):
    he = sample["he_sample"]
    ce = sample["ce_sample"]

    if COMPUTE_COLOR_SEGMENTATION:
        he_quick = apply_color_segmentation_quick_per_blocks(he)
        print("He finished!")
        ce_quick = apply_color_segmentation_quick_per_blocks(ce)
        print("Ce finished!")

        imsave(output_directory + "he_segmented" + str(i) + ".png", he_quick)
        imsave(output_directory + "ce_segmented" + str(i) + ".png", ce_quick)
    else:
        he_quick = imread(output_directory + "he_segmented" + str(i) + ".png")
        ce_quick = imread(output_directory + "ce_segmented" + str(i) + ".png")

    if COMPUTE_POINTS:
        fixed_points, moving_points = compute_registration_points(he_quick, ce_quick)

        pickle.dump(fixed_points, open(output_directory + "fixed_points" + str(i), "wb"))
        pickle.dump(moving_points, open(output_directory + "moving_points" + str(i), "wb"))
    else:
        fixed_points = pickle.load(open(output_directory + "fixed_points" + str(i), "rb"))
        moving_points = pickle.load(open(output_directory + "moving_points" + str(i), "rb"))

    if COMPUTE_TRANSFORM:
        transform = compute_point_registration_transform(fixed_points, moving_points)
        pickle.dump(transform, open(output_directory + "transform" + str(i), "wb"))
    else:
        transform = pickle.load(open(output_directory + "transform" + str(i), "rb"))

    he_registered, ce_mask_with_filled_holes_registered = \
        register_hematoxylin_and_cytokeratin_mask_with_filled_holes(he, ce, transform)

    he_with_ce_mask = superimpose_mask_on_image(he_registered,
                                                ce_mask_with_filled_holes_registered)

    print("Saving results")
    imsave(output_directory + "he_with_ce_mask_filled_holes" + str(i) + ".png", he_with_ce_mask)

t1 = time.time()

print("Running time: ", t1 - t0)


#%%

#Diagnostics
from utils.point_tools import get_array_from_multi_point, show_points_on_image
from utils.image_tools import si

ce_quick_with_points = show_points_on_image(ce_quick, get_array_from_multi_point(moving_points))
he_quick_with_points = show_points_on_image(he_quick, get_array_from_multi_point(fixed_points))

si(he_quick_with_points, filename="he_quick_points.png")
#%%
si(ce_quick_with_points, filename="ce_quick_points.png")
