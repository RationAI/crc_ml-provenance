from skimage import img_as_bool
from scipy import ndimage
import numpy as np
from scipy.spatial import KDTree
from xml.dom import minidom


def mask_border(binary_mask):
    struct = ndimage.generate_binary_structure(2, 2)
    erode = ndimage.binary_erosion(binary_mask, struct)
    border = binary_mask ^ erode
    return img_as_bool(border)


def point_mask_distances(mask, points):
    mask = img_as_bool(mask)
    border = mask_border(mask)
    border_points = np.argwhere(border)
    tree = KDTree(border_points)
    dist, ind = tree.query(points, k=1)
    return dist


def closest_points(mask,points):
    mask = img_as_bool(mask)
    border = mask_border(mask)
    border_points = np.argwhere(border)
    tree = KDTree(border_points)
    dist, ind = tree.query(points, k=1)
    return border_points[ind]


def point_mask_mean_dist(mask, points):
    distances = point_mask_distances(points, mask)
    return np.mean(distances)


def poin_mask_median_dist(mask, points):
    distances = point_mask_distances(points, mask)
    return np.median(distances)


def mean_square(d):
    return np.square(d).mean()


def point_mask_MSE_dist(mask, points):
    distances = point_mask_distances(points, mask)
    return np.square(distances).mean()


def read_annotations(path):
    xml = minidom.parse(path)
    coordinates = xml.getElementsByTagName('Coordinate')
    coor = []
    for c in coordinates:
        X = int(float(c.attributes['X'].value))
        Y = int(float(c.attributes['Y'].value))
        coor.append(np.array([Y,X]))
    return np.asarray(coor)


def evaluate(mask, points):
    distances = point_mask_distances(mask, points)
    result = {
        "MEAN": np.mean(distances),
        "MEDIAN": np.median(distances),
        "MEAN_SQUARE": mean_square(distances)
    }
    return result


