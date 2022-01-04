import numpy as np
import skimage
import histomicstk as htk
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import quickshift
from skimage.exposure import rescale_intensity
from skimage.util import invert

from rationai.data.imreg.muni.utils.parallel_image_processing import process_image_by_blocks

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


def to_shape_white_padding(img, shape):
    new_img = np.ones(shape)
    new_img[0:img.shape[0], 0:img.shape[1]] -= img
    new_img[0:img.shape[0], 0:img.shape[1]] = 1 - new_img[0:img.shape[0], 0:img.shape[1]]
    return new_img


def to_shape_black_padding(img, shape):
    new_img = np.zeros(shape)
    new_img[0:img.shape[0], 0:img.shape[1]] += img
    return new_img


def get_stain_histomics(image, matrix, i, i_o=230):
    im_deconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
        img_as_ubyte(image),
        htk.preprocessing.color_deconvolution.complement_stain_matrix(matrix),
        i_o,
    )
    r = im_deconvolved.Stains[:, :, i]
    res = rescale_intensity(img_as_float(r), out_range=(0, 1))
    return invert(res)


def _get_color_segments_quick(block, kernel_size=3, max_dist=6, ratio=0.5):
    return quickshift(block, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)


def color_segmentation_quick(block_row, block_col, block,
    color_seg_kernel_size, color_seg_max_dist, color_seg_ratio):
    seg = _get_color_segments_quick(block, color_seg_kernel_size, color_seg_max_dist, color_seg_ratio)
    im_colored = skimage.color.label2rgb(seg, block, kind='avg', bg_label=np.nextafter(0, 1))
    return block_row, block_col, im_colored


def apply_color_segmentation_quick_per_blocks(img, color_seg_kernel_size, color_seg_max_dist, color_seg_ratio):
    return process_image_by_blocks(img,
                                   color_segmentation_quick,
                                   (6, 6),
                                   color_seg_kernel_size=color_seg_kernel_size,
                                   color_seg_max_dist=color_seg_max_dist,
                                   color_seg_ratio=color_seg_ratio)


def transform_image_by_shapely_transform(img, transform):
    translate = skimage.transform.SimilarityTransform(translation=(-transform.trans_y, -transform.trans_x))
    translated = skimage.transform.warp(img, translate)

    rotated = skimage.transform.rotate(translated, transform.rotation_angle,
                                       center=(transform.rotation_origin_y, transform.rotation_origin_x))

    return rotated
