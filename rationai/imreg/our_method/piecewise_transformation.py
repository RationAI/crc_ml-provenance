from copy import deepcopy
import numpy as np
import skimage
from scipy import spatial
from sklearn.neighbors import KDTree
from skimage.transform import PiecewiseAffineTransform, warp

from rationai.imreg.our_method.utils.point_tools import get_array_from_multi_point


def multipoint_to_skimage_coords(multipoint):
    coords = get_array_from_multi_point(multipoint)
    return np.flip(coords, axis=1)


def get_point_pairs(moving_points, target, max_dist):
    moving_points = np.asarray(moving_points)
    target = np.asarray(target)
    mp = deepcopy(moving_points)
    kdt = KDTree(target, leaf_size=10)
    dist, ind = kdt.query(mp, k=1)

    fromm = []
    to = []
    for i in range(len(dist)):
        if dist[i] < max_dist:
            fromm.append(mp[i])
            to.append(target[ind[i][0]])

    return fromm, to


def piecewise_transformation(shape, source, target, max_distance):

    fromm, to = get_point_pairs(np.asarray(source), np.asarray(target), max_distance)
    trans = PiecewiseAffineTransform()

    if len(fromm) < 3:
        return None

    mp = list(np.flip(np.asarray(fromm), axis=1))
    tp = list(np.flip(np.asarray(to), axis=1))

    corners = [[0, 0], [shape[1], 0], [0, shape[0]], [shape[1], shape[0]]]

    for a in corners:
        mp.append(a)
        tp.append(a)

    trans.estimate(np.asarray(mp), np.asarray(tp))

    return trans


def bilinear_interpolate(img, coords):
    """
    Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """

    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0+1]
    q12 = img[y0+1, x0]
    q22 = img[y0+1, x0+1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def grid_coordinates(points):
    """
    x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax) for x in range(xmin, xmax)], np.uint32)


# is this used anywhere?
def par(roi_coords, roi_tri_indices, tri_affines, src_img, simplex, result_img):
    coords = roi_coords[roi_tri_indices == simplex]
    num_coords = len(coords)
    out_coords = np.dot(tri_affines[simplex], np.vstack((coords.T, np.ones(num_coords))))
    x, y = coords.T
    result_img[y, x] = bilinear_interpolate(src_img, out_coords)
    #return y, x, bilinear_interpolate(src_img, out_coords)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    import os
    import concurrent.futures
    from functools import partial

    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)
    p = partial(par, roi_coords,roi_tri_indices, tri_affines,src_img,result_img)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        result = executor.map(p, range(len(delaunay.simplices)))
        #for a,b,c in result:
        #    result_img[a,b] = c
    return None


def triangular_affine_matrices(vertices, src_points, dest_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dest_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dest_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]

    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
    # Resultant image will not have an alpha channel
    fromm, to = get_point_pairs(src_points, dest_points, 10)

    if len(fromm) < 3:
        return None

    mp = list(np.flip(np.asarray(fromm), axis=1))
    tp = list(np.flip(np.asarray(to), axis=1))
    shape = dest_shape
    corners = [[0, 0], [shape[1]-1, 0], [0, shape[0]-1], [shape[1]-1, shape[0]-1]]

    for a in corners:
        mp.append(a)
        tp.append(a)

    mp = np.asarray(mp)
    tp = np.asarray(tp)
    mp = mp.astype(int)
    tp = tp.astype(int)

    src_img = src_img[:, :]

    rows, cols = dest_shape
    result_img = np.zeros((rows, cols), dtype)

    delaunay = spatial.Delaunay(tp)
    tri_affines = np.asarray(list(triangular_affine_matrices(delaunay.simplices, mp, tp)))

    process_warp(src_img, result_img, tri_affines, tp, delaunay)

    return result_img


def transform_image_pwise(image, transform):
    if len(image.shape) > 2:
        a = []
        for i in range(image.shape[2]):
            a.append(warp(image[:, :, i], transform.inverse, order=0))
        return np.stack(a, axis=2)

    return warp(image, transform.inverse)


def transform_points_piecewise(points, transform):
    points = np.flip(points, axis=1)
    points = transform(points)
    return np.flip(points, axis=1)


def transform_image_piecewise(mask, fromm, to):
    tr = piecewise_transformation(mask.shape, fromm, to)
    return transform_image_pwise(mask, tr)
