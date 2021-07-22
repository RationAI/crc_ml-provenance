import numpy as np
import math
from skimage.morphology import disk, label, remove_small_objects, convex_hull_object
from skimage.measure import regionprops
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.filters import median, threshold_otsu
from skimage.draw import line
from rationai.imreg.our_method.utils.point_tools import compute_nn_indices
from rationai.imreg.magic_constants import MIN_DISTANCE_BETWEEN_REGIONS, MAX_DISTANCE_BETWEEN_CENTROIDS, BLUR_RADIUS, \
    MINIMUM_AREA, WHITE_THRESH


def connect_regions(image, reg1, reg2):
    centroid1 = reg1.centroid
    centroid2 = reg2.centroid
    rr, cc = line(int(centroid1[0]), int(centroid1[1]), int(centroid2[0]), int(centroid2[1]))
    image[rr, cc] = 1


def region_distance(reg_props1,reg_props2):
    coords1 = np.asarray(reg_props1.coords)
    coords2 = np.asarray(reg_props2.coords)
    minimum = math.inf
    for c1 in coords1:
        for c2 in coords2:
            dist = np.linalg.norm(c1-c2)
            minimum = min(minimum, dist)
    return minimum


def get_ordered_segments(segments_he, segments_ce):
    he = [x[:2] for x in segments_he]
    ce = [x[:2] for x in segments_ce]

    ind = compute_nn_indices(np.array(he, dtype="int32"),
                             np.array(ce, dtype="int32")).ravel()

    return np.array(segments_he)[ind], np.array(segments_ce)


def segment_slide(image):
    im = rgb2gray(image)
    blurred = median(im, disk(BLUR_RADIUS))
    mask = blurred > WHITE_THRESH
    blurred[mask] = 255
    rescaled = equalize_hist(blurred)
    out = rescaled > threshold_otsu(rescaled)
    out = invert(out)
    out = convex_hull_object(out)


    # AK just temporarily until 'connect regs if circular shape' is implemented
    """
    lab = label(out)
    regs = regionprops(lab)

    for i,reg1 in enumerate(regs):
        if reg1.area < MINIMUM_AREA:
            reg = reg1
            min_distance = math.inf
            for j,reg2 in enumerate(regs):
                if (i!=j) and (np.linalg.norm(np.asarray(reg1.centroid)-np.asarray(reg2.centroid)) < MAX_DISTANCE_BETWEEN_CENTROIDS):
                    dist = region_distance(reg1,reg2)
                    if dist < min_distance:
                        min_distance = dist
                        reg = reg2
            if min_distance < MIN_DISTANCE_BETWEEN_REGIONS:
                connect_regions(out, reg1, reg)
    """

    out = convex_hull_object(out)
    # just a "sisoid" fix; original - MINIMUM_AREA == 1000
    out = remove_small_objects(out, 900)
    return out
