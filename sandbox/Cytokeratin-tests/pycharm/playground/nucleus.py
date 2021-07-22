import numpy as np
from scipy import linalg
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import separate_stains

from utils.image_tools import si, rgb2h, rgb2h_cytokeratin_tuned

#%%
ce_quick = imread('output_12092019-2/' + "ce_segmented" + str(0) + ".png")
#%%

im = rgb2h_cytokeratin_tuned(ce_quick)

si(im)
#%%
optimal_nuclei_centers = []
max_nuclei_centers = []

optimal_number_of_nuclei=300

print("Searching for nuclei in the cytokeratin.")

for thr in [0.1]: #np.linspace(0.1, 0.8, 100):
    mask1 = im >= thr

    mask1_bool = np.array(mask1, dtype="bool")

    labels = label(mask1_bool)
    regs = regionprops(labels)

    nuclei_centers = [region.centroid for region in regs if region.area >= 30 and
                      region.area <= 250]

    if (len(nuclei_centers) > len(max_nuclei_centers)):
        max_nuclei_centers = nuclei_centers

    if (len(nuclei_centers) >= optimal_number_of_nuclei):
        optimal_nuclei_centers = nuclei_centers

    print("For thr = ", str(thr), " we have ", len(nuclei_centers), " nuclei.")
    print("The current max, optimal", len(max_nuclei_centers), len(optimal_nuclei_centers))


# %%
ce_quick = imread('output_12092019-2/' + "ce_segmented" + str(0) + ".png")
c = rgb2hed(ce_quick)
# %%
si(ce_quick, filename="ce.png")
si(c[:,:,0], filename="c.png")
#%%
for thr in [0.4]:#np.linspace(0.1, 0.6, 100):
    mask1 = h > thr

    mask1_bool = np.array(mask1, dtype="bool")

    si(mask1_bool)

    labels = label(mask1_bool)
    regs = regionprops(labels)

    nuclei_centers = [region.centroid for region in regs if region.area >= 20 and
                      region.area <=150]
    print("For thr = ", str(thr), " we have ", len(nuclei_centers), " nuclei.")
