import numpy as np
from skimage.measure import label, regionprops

from magic_constants import CE_NUCLEI_MAX_AREA, CE_NUCLEI_MIN_AREA, \
    CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI, HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI, \
    NUCLEI_SEG_COLOR_THR_MAX, NUCLEI_SEG_COLOR_THR_MIN, NUCLEI_SEG_COLOR_THR_STEPS
from utils.image_tools import rgb2h, rgb2h_cytokeratin_tuned


# %%

def _get_nuclei_centers(im, optimal_number_of_nuclei):
    optimal_nuclei_centers = []
    max_nuclei_centers = []

    print("Searching for nuclei in the cytokeratin.")

    for thr in np.linspace(NUCLEI_SEG_COLOR_THR_MIN, NUCLEI_SEG_COLOR_THR_MAX, NUCLEI_SEG_COLOR_THR_STEPS):
        mask1 = im >= thr

        mask1_bool = np.array(mask1, dtype="bool")

        labels = label(mask1_bool)
        regs = regionprops(labels)

        nuclei_centers = [region.centroid for region in regs if region.area >= CE_NUCLEI_MIN_AREA and
                          region.area <= CE_NUCLEI_MAX_AREA]

        if (len(nuclei_centers) > len(max_nuclei_centers)):
            max_nuclei_centers = nuclei_centers

        if (len(nuclei_centers) >= optimal_number_of_nuclei):
            optimal_nuclei_centers = nuclei_centers

        print("For thr = ", str(thr), " we have ", len(nuclei_centers), " nuclei.")
        print("The current max, optimal", len(max_nuclei_centers), len(optimal_nuclei_centers))

    if (optimal_nuclei_centers != []):
        return optimal_nuclei_centers
    else:
        return max_nuclei_centers


def get_nuclei_centers_hematoxylin(img_quick, optimal_number_of_nuclei=HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI):
    im = rgb2h(img_quick)
    return _get_nuclei_centers(im, optimal_number_of_nuclei)


def get_nuclei_centers_cytokeratin(img_quick, optimal_number_of_nuclei=CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI):
    im = rgb2h_cytokeratin_tuned(img_quick)
    return _get_nuclei_centers(im, optimal_number_of_nuclei)
