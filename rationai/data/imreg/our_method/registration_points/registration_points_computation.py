import numpy as np
from shapely.geometry import MultiPoint

from rationai.data.imreg.our_method.registration_points.segment_nuclei import get_nuclei_centers_hematoxylin, \
    get_nuclei_centers_cytokeratin


def compute_registration_points(he_stain_segmented, ce_stain_segmented):
    optimal_he, max_he, mask_h_1, mask_h_2 = get_nuclei_centers_hematoxylin(he_stain_segmented)
    optimal_ce, max_ce, mask_c_1, mask_c_2 = get_nuclei_centers_cytokeratin(ce_stain_segmented)
    he_nuclei = np.array(optimal_he, dtype="int32")
    ce_nuclei = np.array(optimal_ce, dtype="int32")

    # print("The number of fixed points: ", he_nuclei.shape[0])
    # print("The number of moving points: ", ce_nuclei.shape[0])

    fixed_points = MultiPoint(he_nuclei)
    moving_points = MultiPoint(ce_nuclei)
    fixed_max = MultiPoint(max_he)
    moving_max = MultiPoint(max_ce)

    return fixed_points, moving_points, fixed_max, moving_max, mask_h_1, mask_h_2, mask_c_1, mask_c_2
