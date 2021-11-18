import numpy as np
from memory_profiler import profile
from rationai.data.imreg.muni.point_registration.grid_search_transform import hierarchical_parallel_grid_search_translation
from rationai.data.imreg.muni.point_registration.grid_search_transform import hierarchical_parallel_grid_search_rotation
from rationai.data.imreg.muni.utils.point_tools import transform_points
from rationai.data.imreg.muni.utils.point_tools import combine_transforms
from rationai.data.imreg.muni.utils.point_tools import Point_transform

fp = open("data/imreg/logs/log5-registration.log","w+")


@profile(stream=fp)
def compute_point_registration_transform(fixed_points, moving_points, number_of_angle_steps, angle_steps, number_of_steps_grid_search_exp, number_of_parallel_grids):

    final_transform = Point_transform()
    translation = hierarchical_parallel_grid_search_translation(fixed_points, moving_points, 6, 0, number_of_steps_grid_search_exp, number_of_parallel_grids)
    final_transform = combine_transforms([final_transform, translation])
    moving_points = transform_points(moving_points, translation)

    while(True):
        translation = hierarchical_parallel_grid_search_translation(fixed_points, moving_points, 3, 0, number_of_steps_grid_search_exp, number_of_parallel_grids)
        final_transform = combine_transforms([final_transform, translation])
        moving_points = transform_points(moving_points, translation)

        rotation_origin_x = moving_points.centroid.x or 0
        rotation_origin_y = moving_points.centroid.y or 0

        rotation = hierarchical_parallel_grid_search_rotation(fixed_points, moving_points,
                                                              number_of_angle_steps=number_of_angle_steps,
                                                              angle_step=angle_steps,
                                                              rotation_origin_x=rotation_origin_x,
                                                              rotation_origin_y=rotation_origin_y,
                                                              number_of_parallel_grids=number_of_parallel_grids)

        final_transform = combine_transforms([final_transform, rotation],
                                             rotation_origin_x=rotation_origin_x,
                                             rotation_origin_y=rotation_origin_y)

        moving_points = transform_points(moving_points, rotation)

        if np.isclose(rotation.rotation_angle, 0.0):
            break

    return final_transform
