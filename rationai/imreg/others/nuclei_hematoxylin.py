import os
import math
import numpy as np
import scipy as sp
import histomicstk as htk
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.measure
import skimage.color
# from skimage.color import separate_stains, label2rgb
import concurrent.futures
from functools import partial
from random import randint
from matplotlib.pyplot import show, imshow
from skimage.morphology import remove_small_holes, remove_small_objects, binary_closing, disk
# from skimage.measure import label, regionprops
# from skimage.io import imsave
from skimage.segmentation import felzenszwalb
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu, median
from skimage.restoration import denoise_tv_chambolle
from skimage.util import img_as_float
from scipy.ndimage import median_filter
from shapely.geometry import MultiPoint

from rationai.imreg.our_method.utils.parallel_image_processing import compute_image_features_by_blocks, process_image_by_blocks
from rationai.imreg.magic_constants import NUCLEI_SEG_COLOR_THR_MAX, NUCLEI_SEG_COLOR_THR_MIN, NUCLEI_SEG_COLOR_THR_STEPS,\
    CE_NUCLEI_MIN_AREA, HE_NUCLEI_MIN_AREA, CE_NUCLEI_MAX_AREA,\
    CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI, CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI_FOR_PIECEWISE, \
    HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI, HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI_FOR_PIECEWISE


def roundness(rprop):
    return 4 * math.pi * rprop.area / (rprop.perimeter ** 2)


def round(rprop):
    # ratio = 4 * math.pi * rprop.area / (rprop.perimeter ** 2)
    bbox = rprop.bbox
    width = bbox[3]-bbox[1]
    height = bbox[2] - bbox[0]
    # naf = rprop.area/ratio
    return rprop.area / rprop.convex_area > 0.85 and 30 < rprop.area < 100 and 1/3 < width / height < 3


def nuclei_in_mask(mask):
    im = remove_small_holes(mask, 20, in_place=True)
    im = remove_small_objects(im, 20, in_place=True)
    l = label(im)
    r_props = regionprops(l)
    nuc = []
    for p in r_props:
        if round(p):
            nuc.append(p.centroid)
    return MultiPoint(nuc)


def p2(x, y, image):
    result = felzenszwalb(image,multichannel=False, min_size=30, sigma=0, scale=1.05)
    ret = label2rgb(result, image, kind="avg")
    return x, y, ret


def segment_image(image):
    im = denoise_tv_chambolle(image, weight=0.06)
    cv = process_image_by_blocks(im, p2, (6, 6))
    cv = rescale_intensity(cv, in_range=(0,1))
    imsave("segmented" + str(randint(0,1000)) + ".png", cv)
    return cv


def tomask(mask):
    im = remove_small_holes(mask, 20, in_place=False)
    im = remove_small_objects(im, 20, in_place=False)
    l = label(im)
    r_props = regionprops(l)
    result = np.zeros(mask.shape)
    for p in r_props:
        if round(p):
            coords = tuple(zip(*p.coords))
            result[coords] = 1
    return result


def tm(lab):
    l = lab
    r_props = regionprops(l)
    result = np.zeros(lab.shape)
    i = 0
    for p in r_props:
        if 25 < p.area < 130 and p.area/p.convex_area > 0.8:
            i = i+1
            coords = tuple(zip(*p.coords))
            result[coords] = randint(0,255)
    print(i)
    return result


def compute_points_my_method(stain_he, stain_ce):
    h_opt,h_pw,h_opt_mask,h_pw_mask = nuclei_hem_thresholding2(stain_he, HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI,
                                                               HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI_FOR_PIECEWISE)

    c_opt,c_pw,c_opt_mask,c_pw_mask = nuclei_hem_thresholding2(stain_ce, CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI,
                                                               CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI_FOR_PIECEWISE)

    return h_opt, c_opt, h_pw, c_pw, h_opt_mask, h_pw_mask, c_opt_mask, c_pw_mask

def nuclei_hem_thresholding2(he_stain, optimal_nuclei, optimal_for_pw):
    stain = skimage.img_as_float(he_stain)
    #stain = denoise_tv_chambolle(stain, weight=0.06)
    #stain = rescale_intensity(stain, in_range=(0,1))
    stain = segment_image(stain)
    thresholds =  np.linspace(0.3, 1.0, 120)
    def mas(thresh):
        return stain > thresh
    masks = list(map(mas,thresholds))
    pointsets = None
    nuc = partial(nuclei_in_mask)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        pointsets = executor.map(nuc, masks)
    pointsets = list(pointsets)
    optimal_nuclei_centers = []
    max_nuclei_centers = []
    optimal_for_piecewise = []
    opt_mask_id = 0
    pw_mask_id = 0
    max_mask_id =0
    for i,nuclei_centers in enumerate(pointsets):
        if (len(nuclei_centers) > len(max_nuclei_centers)):
            max_nuclei_centers = nuclei_centers
            max_mask_id = i
        if (len(nuclei_centers) >= optimal_nuclei):
            optimal_nuclei_centers = nuclei_centers
            opt_mask_id = i
        if len(nuclei_centers) >= optimal_for_pw:
            optimal_for_piecewise = nuclei_centers
            pw_mask_id = i
    if not optimal_for_piecewise:
        optimal_for_piecewise, pw_mask_id = max_nuclei_centers, max_mask_id
    if not optimal_nuclei_centers:
        optimal_nuclei_centers, opt_mask_id = max_nuclei_centers, max_mask_id
    return optimal_nuclei_centers, optimal_for_piecewise, tomask(masks[opt_mask_id]), tomask(masks[pw_mask_id])


def blob_nuclei(he_stain,ce_stain):
    """
    Computes points from pair of stains.
    :param he_stain: hematoxylin stain from H&E image
    :param ce_stain: hematoxylin stian from H-DAB image
    :return: MultiPoint,Multipoin
    """
    return nuclei_blob(he_stain, 4000, 3, 7, 30), nuclei_blob(ce_stain, 600, 3, 6, 30)


def nuclei_blob(stain, number, min_r, max_r, min_nucleus_area):
    """
    :param stain: stain where darker color is stain
    :param number: maximal number of nuclei
    :param min_r: minimal nuclei radius
    :param max_r: maximal nuclei radius
    :return: MultiPoint
    """
    stain = skimage.util.invert(stain)
    threshold = threshold_otsu(stain)
    nb = partial(nuc_bl, min_r, max_r, threshold, min_nucleus_area)
    cv = compute_image_features_by_blocks(stain, nb, (6, 6))
    grid_size = cv[1][0][2].shape
    all_points = []

    for lis in cv[0]:
        i = lis[0]
        j = lis[1]
        for point in lis[2]:
            all_points.append([(point[0][0] + i * grid_size[0], point[0][1] + j * grid_size[1]), point[1]])

    all_points.sort(key=lambda x: x[1], reverse=False)
    coordinates = [point[0] for point in all_points]

    if len(coordinates) > number:
        return MultiPoint(coordinates[:number])

    return MultiPoint(coordinates)


def check(p):
    return 25 < p.area < 150 and roundness(p) > 0.8 and p.area / p.convex_area > 0.8


def nuc_bl(min_r, max_r, foreground_threshold, min_nucleus_area, x, y, he_stain):
    """
    :param min_r: Minimal radius
    :param max_r: Maximal radius
    :param foreground_threshold: Threshold for foreground separaation.
    :param min_nucleus_area: Minimal area of nucleus
    :param x: x-coordinate in grid
    :param y: y-coordinate in grid
    :param he_stain: stain
    :return: list of x,y,centroid_of_nucleus
    """

    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        he_stain < foreground_threshold)

    im_nuclei_seg_mask = \
        htk.segmentation.nuclear.detect_nuclei_kofahi(he_stain, im_fgnd_mask, min_r, max_r, min_nucleus_area, 4)

    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

    region_props = skimage.measure.regionprops(im_nuclei_seg_mask)

    stain = gaussian(he_stain)
    stain = median_filter(stain, size=3)

    return x, y, [[region.centroid, stain[int(region.centroid[0]), int(region.centroid[1])]]
                  for region in region_props if check(region)]


def nuclei(he_stain, min_r,max_r,foreground_inten, orig=None):
    im_nuclei_stain = skimage.util.invert(he_stain)
    foreground_threshold = foreground_inten
    im_nuclei_stain = median_filter(im_nuclei_stain, size=(2,2))
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < foreground_threshold)

    im_nuclei_seg_mask = \
        htk.segmentation.nuclear.detect_nuclei_kofahi(im_nuclei_stain, im_fgnd_mask, min_r, max_r, 25, 4)
    # filter out small objects
    min_nucleus_area = 25

    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(im_nuclei_seg_mask)
    im_nuclei_seg_mask = tm(im_nuclei_seg_mask)

    #return np.array([[region.centroid,] for region in objProps ], dtype="int32")

    print('Number of nuclei = ', len(objProps))
    from rationai.imreg.our_method.registration_points.segment_nuclei import round
    # Display results
    plt.figure(figsize=(20, 10))
    im_input = im_nuclei_stain
    plt.subplot(1, 2, 1)
    plt.imshow(skimage.color.label2rgb(im_nuclei_seg_mask, im_nuclei_stain, bg_label=0), origin='lower')

    plt.subplot(1, 2, 2)
    if orig is not None:
        plt.imshow(orig, origin='lower')
    else:
        plt.imshow(im_input)
        plt.xlim([0, im_input.shape[1]])
        plt.ylim([0, im_input.shape[0]])

        for i in range(len(objProps)):
            if (not round(objProps[i])):
                continue

    plt.savefig("/mnt/data/home/toufar/a.png")
    plt.show()
    print("X")

