import numpy as np
import shapely
from scipy.stats import trim_mean
from sklearn.neighbors import KDTree

from utils.image_tools import superimpose_mask_on_image
# shapely point to np array
from utils.parallel_processing import _process_args_parallel


def get_array_from_multi_point(multi_point):
    return np.array(list(multi_point.array_interface()["data"]), dtype="int").reshape(-1, 2)


# distance to the nearest neighbor of moving points over fixed_points
def compute_nn_indices(fixed_points, moving_points):
    tree = KDTree(fixed_points)
    neigh = tree.query(moving_points, 1)[1]
    return neigh


def compute_distances_to_nn(fixed_points, moving_points):
    neigh = compute_nn_indices(fixed_points, moving_points)
    neigh_points = fixed_points[neigh]
    distances = list(map(lambda x: np.linalg.norm(x[0] - x[1]),
                         zip(neigh_points, moving_points)))
    return distances


def compute_average_distance_to_nn(fixed_points, moving_points, trim=0.05):
    distances = compute_distances_to_nn(fixed_points, moving_points)
    return trim_mean(distances, trim)  # sum(distances)/neigh.shape[0]


# Adds points to an image as small crosses
def get_cross(point, length=1, width=0):
    line = [(x, y) for x in range(-width, width + 1)
            for y in range(-length, length + 1)]

    cross = np.array(line + [(y, x) for (x, y) in line])

    return np.array([(point[0] + cx, point[1] + cy) for (cx, cy) in cross])


def show_points_on_image(img, points, length=1, width=0, intensity=1):
    z = np.zeros(img.shape[:2])
    for p in points: z[tuple(zip(*get_cross(p, length, width)))] = intensity

    return superimpose_mask_on_image(img, z)


# %% shapely point transforms
class Point_transform:
    def __init__(self, trans_x, trans_y, rotation_angle=0.0, rotation_origin_x=0, rotation_origin_y=0,
                 scaling_factor=1.0):
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.rotation_angle = rotation_angle
        self.rotation_origin_x = rotation_origin_x
        self.rotation_origin_y = rotation_origin_y
        self.scaling_factor = scaling_factor

    def as_list(self):
        return [self.trans_x, self.trans_y, self.rotation_angle,
                self.rotation_origin_x, self.rotation_origin_y, self.scaling_factor]

    def from_vector(self, trans_x, trans_y, rotation_angle,
                    rotation_origin_x, rotation_origin_y, scaling_factor):
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.rotation_angle = rotation_angle
        self.rotation_origin_x = rotation_origin_x
        self.rotation_origin_y = rotation_origin_y
        self.scaling_factor = scaling_factor

    def to_vector(self):
        return (
            self.trans_x,
            self.trans_y,
            self.rotation_angle,
            self.rotation_origin_x,
            self.rotation_origin_y,
            self.scaling_factor
        )


def vector_to_transform(vector):
    return Point_transform(vector[0], vector[1], vector[2], vector[3])


def combine_transforms(transforms, rotation_origin_x=0, rotation_origin_y=0):
    return Point_transform(
        sum([t.trans_x for t in transforms]),
        sum([t.trans_y for t in transforms]),
        sum([t.rotation_angle for t in transforms]),
        rotation_origin_x,
        rotation_origin_y,
        np.prod([t.scaling_factor for t in transforms])
    )


def transform_points(points, transform):
    translated = shapely.affinity.translate(points, transform.trans_x, transform.trans_y)
    rotated = shapely.affinity.rotate(translated, transform.rotation_angle,
                                      use_radians=False,
                                      origin=(transform.rotation_origin_x, transform.rotation_origin_y))
    scaled = shapely.affinity.scale(rotated, transform.scaling_factor)
    return scaled


# %% grid search point transform

def grid_search_point_transform(transforms_list, fixed_points, moving_points):
    measures = []
    for transform in transforms_list:
        transformed = transform_points(moving_points, transform)

        measure = compute_average_distance_to_nn(get_array_from_multi_point(fixed_points),
                                                 get_array_from_multi_point(transformed))
        measure_info = {"transform": transform, "measure": measure}
        measures.append(measure_info)

    return measures


def parallel_grid_search_point_transform(pool, grids, fixed_points, moving_points):
    m = _process_args_parallel(pool, grid_search_point_transform, grids, fixed_points=fixed_points,
                               moving_points=moving_points)
    result = min(sum(m, []), key=lambda x: x["measure"])
    return result
