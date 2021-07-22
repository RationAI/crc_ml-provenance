import numpy as np
import shapely
from scipy.stats import trim_mean
from sklearn.neighbors import KDTree

from rationai.imreg.our_method.utils.parallel_image_processing import _process_args_parallel


def get_array_from_multi_point(multi_point):
    return np.array(list(multi_point.array_interface()["data"]), dtype="int").reshape(-1, 2)


# distance to the nearest neighbor of moving points over fixed_points
def compute_nn_indices(fixed_points, moving_points):
    tree = KDTree(fixed_points)
    neigh = tree.query(moving_points, 1)[1]
    return neigh


def compute_nn(fixed, moving):
    tree = KDTree(fixed)
    neigh, dist = tree.query(moving, 1)
    return np.array(neigh).ravel(),np.array(dist).ravel()


def compute_distances_to_nn(fixed_points, moving_points):
    """
    Computes distances from moving_points to closest point in fixed_points.
    :param fixed_points: Multipoint
    :param moving_points: Multipoint
    :return: list of floats
    """
    neigh = compute_nn_indices(fixed_points, moving_points)
    neigh_points = fixed_points[neigh]
    distances = list(map(lambda x: np.linalg.norm(x[0] - x[1]),
                         zip(neigh_points, moving_points)))
    return distances


def compute_average_distance_to_nn(fixed_points, moving_points, trim=0.05):
    """
    Computes mean distance of moving_points to closest point in fixed points.
    :param fixed_points: Multipoint
    :param moving_points: Multipoint
    :param trim: Fraction of values small and large values to be ignored.
    :return: float
    """

    distances = compute_distances_to_nn(fixed_points, moving_points)
    return trim_mean(distances, trim)


class Point_transform:
    """
    Class which represents transformation.
    """

    def __init__(self, trans_x=0, trans_y=0, rotation_angle=0.0, rotation_origin_x=0, rotation_origin_y=0,
                 scaling_factor=1.0):
        """
        :param trans_x: float
            Translation in x-direction.
        :param trans_y: float
            Translation in y-direction.
        :param rotation_angle: float
        :param rotation_origin_x: float
        :param rotation_origin_y: float
        :param scaling_factor:
        """
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
    """
    Combines multiple transforms.
    :param transforms: list of Point_transform
    :param rotation_origin_x: int
    :param rotation_origin_y: int
    :return: Point_transform
    """

    return Point_transform(
        sum([t.trans_x for t in transforms]),
        sum([t.trans_y for t in transforms]),
        sum([t.rotation_angle for t in transforms]),
        rotation_origin_x,
        rotation_origin_y,
    )


def transform_points(points, transform):
    """
    Transforms points.
    :param points: Multipoint
    :param transform: Point_transform.
    :return: Multipoint
    """
    translated = shapely.affinity.translate(points, transform.trans_x, transform.trans_y)
    rotated = shapely.affinity.rotate(translated, transform.rotation_angle,
                                      use_radians=False,
                                      origin=(transform.rotation_origin_x, transform.rotation_origin_y))
    scaled = shapely.affinity.scale(rotated, transform.scaling_factor)
    return scaled


def grid_search_point_transform(transforms_list, fixed_points, moving_points):
    """
    Transform moving points using tranformations
     in transformation_list and computes mean distance to nearest neighbour.
    :param transforms_list: List of Point_transform
    :param fixed_points:
    :param moving_points:
    :return: List of dictionaries, with keys "transform" and "measure".
    """
    measures = []
    for transform in transforms_list:
        transformed = transform_points(moving_points, transform)

        measure = compute_average_distance_to_nn(get_array_from_multi_point(fixed_points),
                                                 get_array_from_multi_point(transformed))
        measure_info = {"transform": transform, "measure": measure}
        measures.append(measure_info)

    return measures


def parallel_grid_search_point_transform(pool, grids, fixed_points, moving_points):
    """
    :param pool: ProcessPoolExecutor
    :param grids: list of lists of Point_transform
        Transformation grids.
    :param fixed_points:
    :param moving_points:
    :return: Point_transform
    """
    m = _process_args_parallel(pool, grid_search_point_transform, grids, fixed_points=fixed_points,
                               moving_points=moving_points)
    result = min(sum(m, []), key=lambda x: x["measure"])
    return result
