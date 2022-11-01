import numpy as np
from skimage.color import rgba2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from shapely.affinity import translate
from shapely.geometry import box
from rationai.data.imreg.muni.sample_segmentation.slide_segmentation import segment_slide
from rationai.data.imreg.muni.sample_segmentation.ignore_annotation import change_multipolygon_annotation_level
from rationai.data.imreg.muni.sample_segmentation.ignore_annotation import draw_multipolygon
from rationai.data.imreg.muni.utils.point_tools import compute_nn


def _adjust_coord_to_level(coord, level, reverse=False):
    """
    Change resolution level of coordinates.
    :param coord: float or int
    :param level: int
    :param reverse: Bool
    :return: float or int
    """
    if reverse:
        return coord * (2 ** level)
    else:
        return coord // (2 ** level)


def pad_bounding_box(bbox, n):
    """
    Enlarge bounding box by n pixels.
    :param bbox: tuple of int
    :param n: int
    :return: tuple of int
    """
    return bbox[0] - n, bbox[1] - n, bbox[2] + n, bbox[3] + n


def segment_binary_objects(im, min_area):
    """
    Returns bounding-boxes of objects in binary mask which are larger than min_area.
    :param im: ndarray image
    :param min_area: int
    :return: list of tuples
    """
    cleared = clear_border(im)
    label_image = label(cleared)
    return [region.bbox for region in regionprops(label_image) if (region.area >= min_area)]


def get_segments(im_opensl, segmentation_level,
                 minimum_sample_area, ignore_annotation,
                 blur_radius, white_thresh, minimum_area):
    im = np.array(im_opensl.read_region((0, 0), segmentation_level,
                                        im_opensl.level_dimensions[segmentation_level]))
    if ignore_annotation is not None:
        ignore_annotation = change_multipolygon_annotation_level(ignore_annotation, 0, segmentation_level)

    i = rgba2rgb(im, background=(1.0, 1.0, 1.0)) * 255
    mask = segment_slide(i, blur_radius, white_thresh, minimum_area)

    if ignore_annotation is not None:
        draw_multipolygon(ignore_annotation, mask, False, 1)
    return segment_binary_objects(mask, minimum_sample_area)


def bbox_to_openslide_coordinates(bbox, level_from, level_to):
    """
    Generates openslide coordinates from bounding box and changes resolution level.
    :param bbox: tuple
    :param level_from: int
    :param level_to: int
    :return: tuple
        x,y,width,height
    """
    return (_adjust_coord_to_level(bbox[1], level_from, True),
            _adjust_coord_to_level(bbox[0], level_from, True),
            _adjust_coord_to_level(bbox[3] - bbox[1], level_from - level_to, True),
            _adjust_coord_to_level(bbox[2] - bbox[0], level_from - level_to, True))


def get_segment_from_openslide(openslide_img, coord_segmentation_level, segmentation_level,
                               level, ignore_annotations=None):
    """
    :param openslide_img: Openslide image
    :param coord_segmentation_level: list of tuples
        Eat tuple represents bounding box.
    :param segmentation_level: int
        Resolution level of bounding boxes.
    :param level: int
        Resolution level of returned images.
    :param ignore_annotations: Shapely.Multipolygon
    :return: image,Shapely.Multipolygon
         Returns image and interesction of it's bounding-box with ignore_annotation.
    """
    c = np.array(coord_segmentation_level)

    coord_level = bbox_to_openslide_coordinates(c, segmentation_level, level)
    coordinates_smaller = bbox_to_openslide_coordinates(c, segmentation_level, level + 2)

    segment = openslide_img.read_region((coord_level[0], coord_level[1]), level,
                                        (coord_level[2], coord_level[3]))

    segment_smaller = openslide_img.read_region((coordinates_smaller[0], coordinates_smaller[1]), level + 2,
                                                (coordinates_smaller[2], coordinates_smaller[3]))

    segment_np = np.asarray(segment, dtype="float32")
    segment_smaller = np.asarray(segment_smaller, dtype="ubyte")

    new_an = None
    if ignore_annotations is not None:
        ignore_annotations = change_multipolygon_annotation_level(ignore_annotations, 0, level)
        bbox_corners = tuple(_adjust_coord_to_level(x, segmentation_level - level, True) for x in c)
        bbox_pol = box(*bbox_corners)

        if ignore_annotations.intersects(bbox_pol):
            inter = ignore_annotations.intersection(bbox_pol)
            new_an = translate(inter, xoff=-bbox_corners[0], yoff=-bbox_corners[1])
        else:
            new_an = None

    return segment_np, new_an, segment_smaller


def get_ordered_segments(segments_he, segments_ce):
    """
    Find pairs of segments.
    :param segments_he: list of tuples
        y-min,x-min,y-max,x-max
    :param segments_ce: list of tuples
        y-min,x-min,y-max,x-max
    :return: Pair of arrays of segments.
        Two list of segments, in first the order is preserved.
    """
    he = [x[:2] for x in segments_he]
    ce = [x[:2] for x in segments_ce]

    dist, ind = compute_nn(np.array(he, dtype="int32"),
                           np.array(ce, dtype="int32"))

    res_h = np.array(segments_he)[np.array(ind)[np.array(dist) < 50]]
    res_c = np.array(segments_ce)[np.array(dist) < 50]

    return res_h, res_c
