import os
import numpy as np
from concurrent import futures
from functools import partial
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects,remove_small_holes

def mask_sizes(mask, minim, maxim):
    im = remove_small_holes(mask, 20, in_place=False)
    im = remove_small_objects(im, 20, in_place=False)
    l = label(im)
    r_props = regionprops(l)
    result = np.zeros(mask.shape)

    for p in r_props:
        if minim <= p.area <= maxim:
            coords = tuple(zip(*p.coords))
            result[coords] = 1

    return result


def round(rprop):
    bbox = rprop.bbox
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    ratio = width / height

    return rprop.area / rprop.bbox_area > 0.5 \
           and rprop.area / rprop.convex_area > 0.8 \
           and 0.5 < ratio < 2 \
           and 50 < rprop.area < 200


def nuc_in_mask(min_area, max_area, mask):
    labels = label(mask)
    regs = regionprops(labels)
    nuclei_centers = [region.centroid for region in regs if min_area <= region.area <= max_area]
    return nuclei_centers


def _get_nuclei_centers(im, optimal_number_of_nuclei, optimal_number_of_nuclei_for_pieceswise, max_area, min_area,
    nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps):
    thresholds = np.linspace(nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps)

    def mas(thresh):
        mask = im > thresh
        mask1 = np.array(mask, dtype="bool")
        return mask1

    masks = list(map(mas, thresholds))
    p = partial(nuc_in_mask, min_area, max_area)

    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        pointsets = executor.map(p, masks)

    pointsets = list(pointsets)
    optimal_nuclei_centers = []
    max_nuclei_centers = []
    optimal_for_piecewise = []
    opt_mask_id = 0
    pw_mask_id = 0
    max_mask_id = 0

    for i, nuclei_centers in enumerate(pointsets):
        if len(nuclei_centers) > len(max_nuclei_centers):
            max_nuclei_centers = nuclei_centers
            max_mask_id = i
        if len(nuclei_centers) >= optimal_number_of_nuclei:
            optimal_nuclei_centers = nuclei_centers
            opt_mask_id = i
        if len(nuclei_centers) >= optimal_number_of_nuclei_for_pieceswise:
            optimal_for_piecewise = nuclei_centers
            pw_mask_id = i

    from skimage.io import imsave
    from skimage.util import img_as_float

    if not optimal_for_piecewise:
        optimal_for_piecewise, pw_mask_id = max_nuclei_centers, max_mask_id

    if len(optimal_nuclei_centers) > 0:
        return optimal_nuclei_centers, optimal_for_piecewise, \
                mask_sizes(masks[opt_mask_id], min_area, max_area), mask_sizes(masks[pw_mask_id], min_area, max_area)

    return max_nuclei_centers, optimal_for_piecewise,\
           mask_sizes(masks[max_mask_id], min_area, max_area), mask_sizes(masks[pw_mask_id], min_area, max_area)


def get_nuclei_centers_hematoxylin(stain, optimal_number_of_nuclei, hemtoxylin_optimal_number_nuclei_for_piecewise,
    ce_nuclei_max_area, ce_nuclei_min_area, nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps):
    return _get_nuclei_centers(stain, optimal_number_of_nuclei, hemtoxylin_optimal_number_nuclei_for_piecewise,
        ce_nuclei_max_area, ce_nuclei_min_area, nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps)


def get_nuclei_centers_cytokeratin(stain, optimal_number_of_nuclei, cytokeratin_optimal_number_nuclei_for_piecewise,
    ce_nuclei_max_area, ce_nuclei_min_area, nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps):
    return _get_nuclei_centers(stain, optimal_number_of_nuclei, cytokeratin_optimal_number_nuclei_for_piecewise,
        ce_nuclei_max_area, ce_nuclei_min_area, nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps)
