import numpy as np
from shapely.geometry import MultiPoint

from rationai.data.imreg.muni.registration_points.segment_nuclei import get_nuclei_centers_hematoxylin
from rationai.data.imreg.muni.registration_points.segment_nuclei import get_nuclei_centers_cytokeratin


def compute_registration_points(he_stain_segmented, ce_stain_segmented,
    hemtoxylin_optimal_number_nuclei_for_piecewise,
    cytokeratin_optimal_number_nuclei_for_piecewise,
    ce_nuclei_max_area, ce_nuclei_min_area,
    nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps):
    optimal_he, max_he, mask_h_1, mask_h_2 = get_nuclei_centers_hematoxylin(he_stain_segmented, hemtoxylin_optimal_number_nuclei_for_piecewise, hemtoxylin_optimal_number_nuclei_for_piecewise,
    ce_nuclei_max_area, ce_nuclei_min_area, nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps)

    optimal_ce, max_ce, mask_c_1, mask_c_2 = get_nuclei_centers_cytokeratin(ce_stain_segmented, cytokeratin_optimal_number_nuclei_for_piecewise, cytokeratin_optimal_number_nuclei_for_piecewise,
    ce_nuclei_max_area, ce_nuclei_min_area, nuclei_seg_color_thr_min, nuclei_seg_color_thr_max, nuclei_seg_color_thr_steps)

    he_nuclei = np.array(optimal_he, dtype="int32")
    ce_nuclei = np.array(optimal_ce, dtype="int32")

    fixed_points = MultiPoint(he_nuclei)
    moving_points = MultiPoint(ce_nuclei)
    fixed_max = MultiPoint(max_he)
    moving_max = MultiPoint(max_ce)

    return fixed_points, moving_points, fixed_max, moving_max, mask_h_1, mask_h_2, mask_c_1, mask_c_2
