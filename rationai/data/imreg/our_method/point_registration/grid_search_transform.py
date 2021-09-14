import math
from concurrent.futures import ProcessPoolExecutor
from memory_profiler import profile

from rationai.data.imreg.magic_constants import NUMBER_OF_STEPS_GRID_SEARCH_EXP, NUMBER_OF_PARALLEL_GRIDS, \
    TOP_STEP_SIZE_GRID_EXP, BOT_STEP_SIZE_GRID_EXP
from rationai.data.imreg.our_method.utils.point_tools import Point_transform, combine_transforms, transform_points,\
    parallel_grid_search_point_transform
from rationai.data.imreg.our_method.utils.utils_generic import list_to_chunks

fp = open("data/imreg/logs/log5-translation_grid.log", "w+")


@profile(stream=fp)
def get_transform_grids(size, min_size_exp=NUMBER_OF_STEPS_GRID_SEARCH_EXP):
    """
    Generates translation grids.
    :param size: size of grid
    :param min_size_exp: minimal size of grids
    :return: list of list of Point_transform
    """

    return [[Point_transform(x, y)
             for x in range(-2 ** (min_size_exp + size), 2 ** (min_size_exp + size), 2 ** size)
             for y in chunk
             ] for chunk in
            list_to_chunks(range(-2 ** (min_size_exp + size), 2 ** (min_size_exp + size), 2 ** size),
                           NUMBER_OF_PARALLEL_GRIDS)]


fp = open("data/imreg/logs/log5-rotation_grid.log", "w+")


@profile(stream=fp)
def get_angle_grids(number_of_angle_steps, angle_step, rotation_origin_x, rotation_origin_y):
    """
    Generates angle grid for gradient descent.
    :param number_of_angle_steps: int
    :param angle_step: float
    :param rotation_origin_x: int
    :param rotation_origin_y: int
    :return: list of lists of Point_transform
    """
    chunk_size = math.ceil(number_of_angle_steps / NUMBER_OF_PARALLEL_GRIDS)

    grid = [[Point_transform(0, 0, angle_step * x, rotation_origin_x=rotation_origin_x,
                             rotation_origin_y=rotation_origin_y) for x in range(u * chunk_size,
                                                                                 (u + 1) * chunk_size)]
            for u in range(0, NUMBER_OF_PARALLEL_GRIDS)]

    g = [[pt for pt in ptset] + [Point_transform(0, 0, -pt.rotation_angle, rotation_origin_x=rotation_origin_x,
                                                 rotation_origin_y=rotation_origin_y)
                                 for pt in ptset] for ptset in grid]
    return g


fp = open("data/imreg/logs/log5-translation.log", "w+")


@profile(stream=fp)
def hierarchical_parallel_grid_search_translation(fixed_points, moving_points,
                                                  top_level=TOP_STEP_SIZE_GRID_EXP,
                                                  bot_level=BOT_STEP_SIZE_GRID_EXP):
    """
    Computes transformation which minimizes mean distance
        from moving_points to closest point from fixed_points, using gradient descent.
    :param fixed_points: Multipoint
    :param moving_points: Multipoint
    :param top_level: int
        Detemines maximum grid size, which is computed as a * 2 ^ top_level
    :param bot_level: int
        Detemines minimum grid size, which is computed as a * 2 ^ bot_level
    :return: Point_transform

    """
    transforms = []
    pool = ProcessPoolExecutor()

    for size in range(top_level, bot_level, -1):
        result = parallel_grid_search_point_transform(pool, [(g,) for g in get_transform_grids(size)],
                                                      fixed_points,
                                                      moving_points)

        best_transform = result["transform"]
        moving_points = transform_points(moving_points, best_transform)
        transforms.append(best_transform)

    return combine_transforms(transforms)


fp = open("data/imreg/logs/log5-rotation.log", "w+")


@profile(stream=fp)
def hierarchical_parallel_grid_search_rotation(fixed_points, moving_points,
                                               number_of_angle_steps,
                                               angle_step,
                                               rotation_origin_x=0,
                                               rotation_origin_y=0):
    """
    Computes rotation which minimizes mean distance
        from moving_points to closest point from fixed_points.
    :param fixed_points:    Multipoint
    :param moving_points:   Multipoint
        Points to transform.
    :param number_of_angle_steps: int
    :param angle_step:  float
        Size of angle step.
    :param rotation_origin_x: float
        x-coordinate for rotation center
    :param rotation_origin_y:
        x-coordinate for rotation center
    :return: Point_transform
    """

    pool = ProcessPoolExecutor()

    result = parallel_grid_search_point_transform(pool, [(g,) for g in
                                                         get_angle_grids(number_of_angle_steps, angle_step,
                                                                         rotation_origin_x,
                                                                         rotation_origin_y)],
                                                  fixed_points,
                                                  moving_points)

    best_transform = result["transform"]
    return best_transform
