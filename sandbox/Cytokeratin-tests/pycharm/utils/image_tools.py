import numpy as np
import skimage
from scipy import linalg
from skimage.color import rgb2hed, rgb2hsv, separate_stains
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.segmentation import quickshift
from skimage.transform import rescale

from utils.parallel_image_processing import process_image_by_blocks, compute_image_features_by_blocks

from magic_constants import COLOR_SEG_KERNEL_SIZE, COLOR_SEG_MAX_DIST, COLOR_SEG_RATIO


def si(im, filename="t.png", sc=1.0):
    skimage.io.imsave(filename, rescale(im, sc))


# resizing

def add_margin_right(img, shape):
    z = np.zeros(shape)
    z[0:img.shape[0], 0:img.shape[1]] += img
    return z


def add_margins(img, x_size, y_size):
    if len(img.shape) == 3:
        z = np.zeros((img.shape[0] + 2 * x_size, img.shape[1] + 2 * y_size, img.shape[2]))
    else:
        z = np.zeros((img.shape[0] + 2 * x_size, img.shape[1] + 2 * y_size))

    z[x_size:img.shape[0] + x_size, x_size:img.shape[1] + x_size] += img
    return z


def to_shape(img, shape):
    new_img = np.zeros(shape)
    new_img[0:img.shape[0], 0:img.shape[1]] += img
    return new_img


# color space transform

rgb_from_hed = np.array([[0.543, 0.690, 0.479],
                         [0.705, -.684, 0.186],
                         [0.305, 0.529, 0.792]])

hed_from_rgb = linalg.inv(rgb_from_hed)


def rgb2h_cytokeratin_tuned(rgb):
    return rescale_intensity(separate_stains(rgb, hed_from_rgb), out_range=(0, 1))[:,:,0]


def rgb2h(im):
    return rescale_intensity(rgb2hed(im[:, :, :3])[:, :, 0], out_range=(0, 1))


def rgb2dab(im):
    return rescale_intensity(rgb2hed(im[:, :, :3])[:, :, 2], out_range=(0, 1))


# mask over rgb image
def superimpose_mask_on_image(i, m):
    return np.stack([np.array(i[:, :, 0], dtype="float32"),
                     np.array(i[:, :, 1], dtype="float32") + np.array(m, dtype="float32"),
                     np.array(i[:, :, 2], dtype="float32")],
                    axis=2)


# color segmentation - parallel

def _get_color_segments_quick(block, kernel_size=3, max_dist=6, ratio=0.5):
    print("KERNEL SIZE:", kernel_size)

    return quickshift(block, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)


def color_segmentation_quick(block_row, block_col, block):
    print("Segment ", block_row, block_col)
    seg = _get_color_segments_quick(block, COLOR_SEG_KERNEL_SIZE, COLOR_SEG_MAX_DIST, COLOR_SEG_RATIO)

    print("Color ", block_row, block_col)
    im_colored = skimage.color.label2rgb(seg, block, kind='avg')
    return block_row, block_col, im_colored


def apply_color_segmentation_quick_per_blocks(img):
    return process_image_by_blocks(img, color_segmentation_quick, (6, 6))


def compute_color_segments_quick(block_row, block_col, block):
    print("Segment ", block_row, block_col)
    return _get_color_segments_quick(block, COLOR_SEG_KERNEL_SIZE, COLOR_SEG_MAX_DIST, COLOR_SEG_RATIO)


def apply_compute_color_segments_quick_per_blocks(img):
    return compute_image_features_by_blocks(img, compute_color_segments_quick, (6, 6))


# thresholding
def single_otsu_threshold(im, comparison):
    thr = threshold_otsu(im)
    return comparison(im, thr)


def double_otsu_threshold(im):
    im_thr1 = im * single_otsu_threshold(im, lambda im, thr: im < thr)
    return single_otsu_threshold(im_thr1, lambda im, thr: im > thr)


def dual_double_otsu_threshold(im):
    im_thr11 = im * (1 - single_otsu_threshold(im, lambda im, thr: im < thr))

    im_thr1 = 1 - np.array(single_otsu_threshold(im_thr11, lambda im, thr: im < thr), dtype="float32")

    i = np.array((im_thr1 * im < 1) & (im_thr1 * im > 0), dtype="float32")

    return i


def transform_image_by_shapely_transform(img, transform):
    translate = skimage.transform.SimilarityTransform(translation=(-transform.trans_y,
                                                                   -transform.trans_x))
    translated = skimage.transform.warp(img, translate)

    rotated = skimage.transform.rotate(translated, transform.rotation_angle, center=(
        transform.rotation_origin_y, transform.rotation_origin_x))

    scaling = skimage.transform.SimilarityTransform(scale=transform.scaling_factor)
    scaled = skimage.transform.warp(rotated, scaling)

    return scaled


def get_non_white_mask_hsv(im_hsv, sat_white_lower, sat_white_upper):
    im_hsv_mask_non_white = np.array((im_hsv[:, :, 1] >= sat_white_lower) &
                                     (im_hsv[:, :, 1] <= sat_white_upper),
                                     dtype="int32")
    return im_hsv_mask_non_white


def get_non_white_mask_rgb(im, sat_white_lower, sat_white_upper):
    return get_non_white_mask_hsv(rgb2hsv(im), sat_white_lower, sat_white_upper)
