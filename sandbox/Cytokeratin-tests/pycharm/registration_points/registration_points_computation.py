import numpy as np
from shapely.geometry import MultiPoint

from registration_points.segment_nuclei import get_nuclei_centers_hematoxylin, \
    get_nuclei_centers_cytokeratin


def compute_registration_points(he_quick, ce_quick):
    he_nuclei = np.array(get_nuclei_centers_hematoxylin(he_quick), dtype="int32")

    ce_nuclei = np.array(get_nuclei_centers_cytokeratin(ce_quick), dtype="int32")

    print("The number of fixed points: ", he_nuclei.shape[0])
    print("The number of moving points: ", ce_nuclei.shape[0])

    fixed_points = MultiPoint(he_nuclei)
    moving_points = MultiPoint(ce_nuclei)

    return fixed_points, moving_points
