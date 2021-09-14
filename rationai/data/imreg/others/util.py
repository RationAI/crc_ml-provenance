import numpy as np
from skimage import img_as_ubyte
import histomicstk as htk
from skimage.draw import circle_perimeter


def compute_stain_matrix_he_histomics(image):
    im_input = img_as_ubyte(image)

    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',  # cytoplasm stain
              'null']  # set to null if input contains only two stains

    # create stain matrix
    stain_matrix = np.array([stain_color_map[st] for st in stains]).T

    # create initial stain matrix
    stain_matrix = stain_matrix[:, :2]

    # Compute stain matrix adaptively
    sparsity_factor = 0.5

    i_0 = 230
    im_sda = htk.preprocessing.color_conversion.rgb_to_sda(im_input, i_0)
    stain_matrix = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(im_sda, stain_matrix, sparsity_factor)

    return stain_matrix


def compute_stain_matrix_hdab_histomics(image):
    im_input = img_as_ubyte(image)

    # create stain to color map
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # print('stain_color_map:', stain_color_map, sep='\n')

    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'dab',  # cytoplasm stain
              'null']  # set to null if input contains only two stains

    # create stain matrix
    stain_matrix = np.array([stain_color_map[st] for st in stains]).T

    # create initial stain matrix
    stain_matrix = stain_matrix[:, :2]

    # Compute stain matrix adaptively
    sparsity_factor = 0.5

    i_0 = 230
    im_sda = htk.preprocessing.color_conversion.rgb_to_sda(im_input, i_0)
    stain_matrix = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
        im_sda, stain_matrix, sparsity_factor,
    )
    return stain_matrix


def draw_points(coordinates, shape):
    mask = np.zeros(shape)

    for c in coordinates:
        rr, cc = circle_perimeter(int(c[0]), int(c[1]), 3)
        for a, b in zip(rr, cc):
            mask[int(a), int(b)] = 1

    return np.stack([np.zeros(mask.shape),
                     np.zeros(mask.shape),
                     mask,
                     mask],
                    axis=2)
