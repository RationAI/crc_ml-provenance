import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, diamond

from magic_constants import HOLES_MAX_AREA, HOLES_H_PROPORTION, HOLES_MIN_AREA
from utils.image_tools import double_otsu_threshold, rgb2dab, dual_double_otsu_threshold, rgb2h
from utils.parallel_image_processing import process_image_by_blocks


def get_regions_in_cytokeratin_mask(cytokeratin_mask):
    dual_cytokeratin_mask = binary_closing(1 - cytokeratin_mask.astype(int), diamond(4))
    labels = label(dual_cytokeratin_mask)
    regs = regionprops(labels)
    return regs


def compute_cytokeratin_mask_holes(cytokeratin_mask, hematoxylin_mask,
                                   hole_hematoxylin_minimum_proportion,
                                   hole_min_area, hole_max_area, take_all_small_holes=True):
    regs = get_regions_in_cytokeratin_mask(cytokeratin_mask)

    holes = np.zeros(cytokeratin_mask.shape)
    for region in regs:
        if region.area <= hole_max_area:
            coords = tuple(zip(*region.coords))
            if region.area < hole_min_area:
                if take_all_small_holes:
                    holes[coords] = 1
            else:
                H_values = (1 - hematoxylin_mask)[coords]
                meanH = np.mean(H_values)
                if meanH >= hole_hematoxylin_minimum_proportion:
                    holes[coords] = 1
    return holes


# parallel
def _get_cytokeratin_mask_holes(i, j, mask):
    cytokerating_mask = mask[:, :, 0]
    hematoxylin_mask = mask[:, :, 1]

    print("Filling holes in block ", i, j)

    cytokeratin_holes = compute_cytokeratin_mask_holes(cytokerating_mask, hematoxylin_mask,
                                                       HOLES_H_PROPORTION,
                                                       HOLES_MIN_AREA,
                                                       HOLES_MAX_AREA,
                                                       True)

    return i, j, cytokeratin_holes


def get_cytokeratin_mask(img):
    return dual_double_otsu_threshold(rgb2dab(img))


def get_hematoxylin_mask(img):
    return double_otsu_threshold(rgb2h(img))


def get_cytokeratin_mask_holes(cytokeratin_mask, hematoxylin_mask):
    return process_image_by_blocks(
        np.stack([
            cytokeratin_mask,
            hematoxylin_mask.reshape(cytokeratin_mask.shape)],
            axis=2),
        _get_cytokeratin_mask_holes,
        (6, 6))
