from concurrent.futures import ProcessPoolExecutor

from magic_constants import NUMBER_OF_STEPS_GRID_SEARCH_EXP, NUMBER_OF_PARALLEL_GRIDS, \
    TOP_STEP_SIZE_GRID_EXP, BOT_STEP_SIZE_GRID_EXP, STOPPING_BAD_SUFFIX_LENGTH, \
    LOCAL_SEARCH_NUMBER_OF_STEPS, LOCAL_SEARCH_STEP_SIZE
from utils.point_tools import Point_transform
from utils.point_tools import combine_transforms, transform_points, parallel_grid_search_point_transform
from utils.utils_generic import list_to_chunks


# %%
def get_transform_grids(size, min_size_exp=NUMBER_OF_STEPS_GRID_SEARCH_EXP):
    return [[Point_transform(x, y)
             for x in range(-2 ** (min_size_exp + size), 2 ** (min_size_exp + size), 2 ** size)
             for y in chunk] for chunk in
            list_to_chunks(range(-2 ** (min_size_exp + size), 2 ** (min_size_exp + size), 2 ** size),
                           NUMBER_OF_PARALLEL_GRIDS)]


def get_transform_grids_relative(old_transform, number, size, angles):
    return list_to_chunks(
        [Point_transform(x + old_transform.trans_x, y + old_transform.trans_y,
                         rotation_angle=a + old_transform.rotation_angle,
                         rotation_origin_x=old_transform.rotation_origin_x,
                         rotation_origin_y=old_transform.rotation_origin_y)
         for x in range(-number * size, number * size, size)
         for y in range(-number * size, number * size, size)
         for a in angles], NUMBER_OF_PARALLEL_GRIDS)


def hierarchical_parallel_grid_search_translation(fixed_points, moving_points,
                                                  top_level=TOP_STEP_SIZE_GRID_EXP,
                                                  bot_level=BOT_STEP_SIZE_GRID_EXP):
    transforms = []
    pool = ProcessPoolExecutor()

    for size in range(top_level, bot_level, -1):
        result = parallel_grid_search_point_transform(pool, [(g,) for g in get_transform_grids(size)],
                                                      fixed_points,
                                                      moving_points)

        best_transform = result["transform"]
        print(best_transform.as_list())
        print("Transforming moving")
        moving_points = transform_points(moving_points, best_transform)
        print("Append best")
        transforms.append(best_transform)

    return combine_transforms(transforms)


def decide_to_stop(results_list):
    if len(results_list) >= STOPPING_BAD_SUFFIX_LENGTH:
        last_n_minimum = min(results_list[-(STOPPING_BAD_SUFFIX_LENGTH - 1):], key=lambda x: x["measure"])["measure"]
        minus_n_value = results_list[-STOPPING_BAD_SUFFIX_LENGTH]["measure"]
        return minus_n_value <= last_n_minimum
    else:
        return False


def locally_optimal_rotation_by_angle(fixed_points, moving_points, transform, angle, pool):
    result = parallel_grid_search_point_transform(pool,
                                                  [(g,) for g in get_transform_grids_relative(
                                                      transform, LOCAL_SEARCH_NUMBER_OF_STEPS,
                                                      LOCAL_SEARCH_STEP_SIZE,
                                                      [angle])],
                                                  fixed_points,
                                                  moving_points)
    print(result["transform"].as_list(), result["measure"])
    return result


def find_optimal_rotation_single_direction(fixed_points, moving_points, transform, single_turn_angle, pool,
                                           number_of_turns):
    results = []
    for i in range(0, number_of_turns):
        result = locally_optimal_rotation_by_angle(fixed_points, moving_points, transform, single_turn_angle, pool)
        transform = result["transform"]
        results.append(result)
        if (decide_to_stop(results)):
            break

    return results


def grid_search_translation_and_angle(fixed_points, moving_points, single_turn_angle,
                                      number_of_turns):
    pool = ProcessPoolExecutor()
    min_transform = Point_transform(0, 0, rotation_angle=0,
                                    rotation_origin_x=moving_points.centroid.x,
                                    rotation_origin_y=moving_points.centroid.y,
                                    scaling_factor=1.0)

    results_positive = find_optimal_rotation_single_direction(fixed_points, moving_points, min_transform,
                                                              single_turn_angle, pool, number_of_turns)

    results_negative = find_optimal_rotation_single_direction(fixed_points, moving_points, min_transform,
                                                              -single_turn_angle, pool, number_of_turns)

    results = results_negative + results_positive

    best_result = min(results, key=lambda x: x["measure"])

    return best_result["transform"]
