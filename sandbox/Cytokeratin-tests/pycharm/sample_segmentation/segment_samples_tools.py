import numpy as np
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation
from skimage.segmentation import clear_border

from utils.image_tools import get_non_white_mask_hsv
from utils.point_tools import compute_nn_indices


# Returns a part of an image given coordinates
def get_subimage(im, coords):
    return im[coords[0]:coords[2], coords[1]:coords[3]]


def _adjust_coord_to_level(coord, level, reverse=False):
    if reverse:
        return coord * (2 ** level)
    else:
        return coord // (2 ** level)


def adjust_bbox_to_crop(bbox, pre_crop_coords, segmentation_level):
    return (
        bbox[0] + _adjust_coord_to_level(pre_crop_coords[0], segmentation_level),
        bbox[1] + _adjust_coord_to_level(pre_crop_coords[1], segmentation_level),
        bbox[2] + _adjust_coord_to_level(pre_crop_coords[0], segmentation_level),
        bbox[3] + _adjust_coord_to_level(pre_crop_coords[1], segmentation_level)
    )


# Returns a list of objects larger than min_area

def segment_binary_objects(im, min_area, pre_crop_coords, segmentation_level):
    cleared = clear_border(im)
    label_image = label(cleared)

    return [adjust_bbox_to_crop(region.bbox, pre_crop_coords, segmentation_level) for
            region in regionprops(label_image) if (region.area >= min_area)]


def get_segments(im_opensl, segmentation_level,
                 minimum_sample_area, sat_white_lower, sat_white_upper, rubbish_hue_lower,
                 closing_diam_before_segmentation, pre_crop_coords):
    im_seg_level = np.array(im_opensl.read_region((0, 0), segmentation_level,
                                                  im_opensl.level_dimensions[segmentation_level]),
                            dtype="float32")[:, :, :3]

    im = im_seg_level[_adjust_coord_to_level(pre_crop_coords[0], segmentation_level):
                      _adjust_coord_to_level(pre_crop_coords[2], segmentation_level),
         _adjust_coord_to_level(pre_crop_coords[1], segmentation_level):
         _adjust_coord_to_level(pre_crop_coords[3], segmentation_level),
         :]

    im_hsv = rgb2hsv(im)

    im_hsv_mask_non_white = get_non_white_mask_hsv(im_hsv, sat_white_lower, sat_white_upper)

    im_hsv_mask_rubbish_by_hue = np.array((im_hsv[:, :, 0] >= rubbish_hue_lower),
                                          dtype="int32")

    res = dilation(im_hsv_mask_non_white * im_hsv_mask_rubbish_by_hue,
                   selem=disk(closing_diam_before_segmentation))

    return segment_binary_objects(res, minimum_sample_area, pre_crop_coords,
                                  segmentation_level), \
           im_hsv_mask_non_white


# Returns an image from a slide for coordinates returned by get_segments

def get_segment_from_openslide(openslide_img, coord_segmentation_level, segmentation_level,
                               level):
    c = np.array(coord_segmentation_level)

    coord_level = (_adjust_coord_to_level(c[1], segmentation_level, True),
                   _adjust_coord_to_level(c[0], segmentation_level, True),
                   _adjust_coord_to_level(c[3] - c[1], segmentation_level - level, True),
                   _adjust_coord_to_level(c[2] - c[0], segmentation_level - level, True))

    segment = openslide_img.read_region((coord_level[0], coord_level[1]), level,
                                        (coord_level[2], coord_level[3]))

    segment_np = np.asarray(segment)

    return segment_np


def get_ordered_segments(segments_he, segments_ce):
    he = [x[:2] for x in segments_he]
    ce = [x[:2] for x in segments_ce]

    ind = compute_nn_indices(np.array(he, dtype="int32"),
                             np.array(ce, dtype="int32")).ravel()

    return np.array(segments_he)[ind], np.array(segments_ce)
