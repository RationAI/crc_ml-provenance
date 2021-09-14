import numpy as np
from memory_profiler import profile
from rationai.data.imreg.our_method.point_registration.grid_search_transform import hierarchical_parallel_grid_search_translation, \
    hierarchical_parallel_grid_search_rotation
from rationai.data.imreg.our_method.utils.point_tools import transform_points, combine_transforms, Point_transform
from rationai.data.imreg.magic_constants import NUMBER_OF_ANGLE_STEPS, ANGLE_STEP

fp = open("data/imreg/logs/log5-registration.log","w+")


@profile(stream=fp)
def compute_point_registration_transform(fixed_points, moving_points):

    final_transform = Point_transform()
    translation = hierarchical_parallel_grid_search_translation(fixed_points, moving_points, 6, 0)
    final_transform = combine_transforms([final_transform, translation])
    moving_points = transform_points(moving_points, translation)

    while(True):
        translation = hierarchical_parallel_grid_search_translation(fixed_points, moving_points, 3, 0)
        final_transform = combine_transforms([final_transform, translation])
        moving_points = transform_points(moving_points, translation)

        rotation_origin_x = moving_points.centroid.x
        rotation_origin_y = moving_points.centroid.y

        rotation = hierarchical_parallel_grid_search_rotation(fixed_points, moving_points,
                                                              number_of_angle_steps=NUMBER_OF_ANGLE_STEPS,
                                                              angle_step=ANGLE_STEP,
                                                              rotation_origin_x=rotation_origin_x,
                                                              rotation_origin_y=rotation_origin_y)

        final_transform = combine_transforms([final_transform, rotation],
                                             rotation_origin_x=rotation_origin_x,
                                             rotation_origin_y=rotation_origin_y)

        moving_points = transform_points(moving_points, rotation)

        if np.isclose(rotation.rotation_angle, 0.0):
            break

    return final_transform
