import numpy as np
from skimage import img_as_bool, img_as_float
from skimage.util import invert
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_minimum, threshold_isodata
from skimage.filters.rank import enhance_contrast
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

from rationai.imreg.magic_constants import HOLES_MAX_AREA, HOLES_MIN_AREA, MASK_MIN_AREA


def mask_thresholded(stain, thresholding):
    thresh = thresholding(stain)
    mask = stain > thresh
    image = img_as_bool(mask)
    image = binary_closing(image, disk(3))
    image = remove_small_objects(image, MASK_MIN_AREA)
    image = remove_small_holes(image, HOLES_MIN_AREA)
    return image


def cyt_mask_minimum(stain):
    return mask_thresholded(stain, threshold_minimum)


def cyt_mask_isodata(stain):
    return mask_thresholded(stain, threshold_isodata)


def create_cytokeratin_mask(stain):
    """
    Creates binary mask from grayscale image using threshold_minimum, if this
    thresholding fails, threshold isodata is used. Small objects and holes are removed.
    :param stain: ndarray
        Image with shape NxM.
    :return: ndarray
        Binary mask with shape NxM.
    """

    try:
        thresh = threshold_minimum(stain)
        # print("THRESHOLD MINUMUM")
    except:
        thresh = threshold_isodata(stain)
        # print("THRESHOLD ISODATA")
    mask = stain > thresh
    mask = binary_closing(mask, disk(3))
    remove_small_objects(mask, MASK_MIN_AREA, in_place=True)
    remove_small_holes(mask, HOLES_MIN_AREA, in_place=True)
    return mask


def holes_in_mask(mask, max_size):
    """
    :param mask: ndarray
        binary mask
    :param max_size: int
        maximal size of holes
    :return: array of regionprops of holes
    """
    inverted = invert(mask)
    lb = label(inverted)
    regions = np.asarray(regionprops(lb))
    return regions[[r.area < max_size for r in regions]]


def hole_mask(mask):
    """
    :param mask: ndarray
        Binary mask.
    :return: ndarray
        Binary mask of holes in mask.
    """
    h_mask = np.zeros(mask.shape, dtype=bool)
    regionprops = holes_in_mask(mask, HOLES_MAX_AREA)
    for r in regionprops:
        coords = tuple(zip(*r.coords))
        h_mask[coords] = 1
    return h_mask


def fill_holes(mask, hematoxylin_mask):
    """
    Fills holes in mask. Fills parts of holes which intersect hematoxylin_mask,
    then fills remaining small holes.

    :param mask: ndarray
    :param hematoxylin_mask: ndarray
    :return: ndarray
        Mask with filled holes.
    """
    mask = img_as_bool(mask)
    mask = binary_closing(mask, disk(2))
    hole_m = hole_mask(mask)
    hole_m = remove_small_holes(hole_m, 20, in_place=False)
    out = np.logical_or(mask, np.logical_and(hole_m, hematoxylin_mask))
    out = binary_closing(out, disk(3))
    out = remove_small_holes(out, HOLES_MIN_AREA, in_place=True)
    return out


def create_he_mask(hematoxylin_stain):
    """
    Creates binary mask of hematoxylin stain using Otsu method.
    :param hematoxylin_stain: ndarray NxM
    :return: ndarray
        Binary mask.
    """
    mask = hematoxylin_stain > threshold_otsu(hematoxylin_stain)
    mask = binary_closing(mask, disk(2))
    mask = remove_small_holes(mask, 20)
    mask = remove_small_objects(mask, 20)
    return mask


def mask_remove_he_background(cytokeratin_mask, he_image, bg_thresh=195):
    """
    Removes regions of a mask which are considered to be a background of HE image
    :param cytokeratin_mask: ndarray
    :param he_image: ndarray
    :param bg_thresh: int [0-255] background threshold
    :return: ndarray
        Binary mask
    """
    gray = rgb2gray(he_image)
    enhanced = enhance_contrast(gray, disk(3))

    he_to_remove = (enhanced > bg_thresh) * 255
    mask = cytokeratin_mask - he_to_remove
    mask = np.clip(mask, a_min=0, a_max=255)

    mask = binary_closing(mask, disk(3))
    mask = remove_small_objects(mask, MASK_MIN_AREA)
    mask = remove_small_holes(mask, 150)  # todo: magic number
    return mask


def green_mask(mask):
    """
    Generates green mask, with transparency
    :param mask: ndarray NxM
    :return: ndarray NxMx4
    """
    mask = img_as_float(mask)
    out = np.stack([np.zeros(mask.shape), mask, np.zeros(mask.shape), mask*0.55], axis=2)
    return out


def cyan_mask(mask):
    """
    Generates cyan mask, with transparency
    :param mask: ndarray NxM
    :return: ndarray NxMx4
    """
    mask = img_as_float(mask)
    out = np.stack([np.zeros(mask.shape), mask, mask, mask*0.55], axis=2)
    return out


def rgba_mask(mask):
    out = np.stack([np.full(mask.shape,255,dtype=np.uint8), mask, np.zeros(mask.shape), mask], axis=2)
    return out
