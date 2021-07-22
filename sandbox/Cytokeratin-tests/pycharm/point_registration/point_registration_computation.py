from point_registration.grid_search_transform import hierarchical_parallel_grid_search_translation, \
    grid_search_translation_and_angle
from utils.point_tools import transform_points, combine_transforms

from magic_constants import ANGLE_STEP, NUMBER_OF_ANGLE_STEPS


def compute_point_registration_transform(fixed_points, moving_points):
    translation = hierarchical_parallel_grid_search_translation(fixed_points, moving_points)
    moving_points_translated = transform_points(moving_points, translation)
    rotation = grid_search_translation_and_angle(fixed_points, moving_points_translated,
                                                 ANGLE_STEP,
                                                 NUMBER_OF_ANGLE_STEPS)
    final_transform = combine_transforms([translation, rotation],
                                         rotation_origin_x=rotation.rotation_origin_x,
                                         rotation_origin_y=rotation.rotation_origin_y)
    return final_transform
