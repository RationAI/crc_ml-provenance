import numpy as np
from xml.dom import minidom
from shapely.geometry import Polygon
from shapely.affinity import scale
from shapely.geometry import MultiPolygon
from skimage.draw import polygon


def polygon_to_annotation(polygon):
    return np.asarray(polygon.exterior.coords)[:-1, :]


def annotation_intersection(annotation1, annotation2):
    pol1 = Polygon(annotation1)
    pol2 = Polygon(annotation2)
    if not pol1.intersects(pol2):
        return None
    inter = pol1.intersection(pol2)
    return polygon_to_annotation(inter)


def annotation_to_mask(annotations, shape):
    mask = np.zeros(shape)
    for a in annotations:
        rr, cc = polygon(a[:, 0], a[:, 1])
        mask[rr, cc] = 1
    return mask


def annotation_to_image(vertices_of_polygon, image, value, channel):
    """

    :param vertices_of_polygon: ndarray Nx2
        Coordinates of polygon vertices
    :param image: ndarray
    :param value: float or int
    :param channel: int
        Channel to modify in case of multichannel images,
    :return: ndarray
        Image where areas from annotations are assigned new value.
    """
    for a in vertices_of_polygon:
        rr, cc = polygon(a[:, 0], a[:, 1])
        for a,b in zip(rr, cc):
            if len(image.shape) == 2:
                image[a, b] = value
            else:
                image[a, b, channel] = value
    return image


def draw_multipolygon(multipolygon, image, value, channel):
    for p in multipolygon.geoms:
        draw_polygon(p, image, value, channel)


def draw_polygon(polygon, image, value, channel):
    global x
    annot = polygon_to_annotation(polygon)
    draw_annotation_on_image(annot, image, value, channel)


def draw_annotation_on_image(annotation, image, value, chanel=0):
    rr, cc = polygon(annotation[:,0], annotation[:,1])
    for a, b in zip(rr, cc):
        if a >= image.shape[0] or b >= image.shape[1]:
            continue
        if len(image.shape) == 2:
            image[a, b] = value
        else:
            image[a, b, chanel] = value
    return image


def annotation_change_level(annotation, source_level, target_level):
    if source_level > target_level:
        return (2 ** (source_level - target_level)) * annotation
    return annotation // (2 ** (target_level - source_level))


def annotation_to_polygon(annotation):
    return Polygon(annotation)


def read_ignore_annotations(path, output_level):
    """
    Read and process annotation is from ASAP.
    :param path: str
        Path to XML file containing annotation
    :param output_level: int
        Output level of processed annotations.
    :return: list of Shapely.Polygon

    """

    xml = minidom.parse(path)
    polygons = xml.getElementsByTagName('Annotation')
    out = []
    for pol in polygons:
        coordinates = pol.getElementsByTagName('Coordinate')
        coords = []
        for c in coordinates:
            x = int(float(c.attributes['X'].value) / 2 ** output_level)
            y = int(float(c.attributes['Y'].value) / 2 ** output_level)
            coords.append(np.array([y, x]))
        out.append(np.asarray(coords))
    return out


def read_ignore_annotation_as_multipolygon(path, output_level):
    """
    Reads input annotations and retuns Multipolygon
    :param path: str
        path to ASAP annotation file
    :param output_level: int
        Output level of processed annotations.
    :return: ShapelyMultipolygon
    """
    annotations = read_ignore_annotations(path, output_level)
    return MultiPolygon([Polygon(p) for p in annotations])


def change_multipolygon_annotation_level(mutipolygon, source_level, target_level):
    """
    Rescales multipolygon.
    :param mutipolygon: Shapely.MultiPolygon
    :param source_level: int
    :param target_level: int
    :return: Shapely.MultiPolygon
    """
    return scale(mutipolygon, 2 ** (source_level-target_level), 2 ** (source_level-target_level), origin=(0, 0))
