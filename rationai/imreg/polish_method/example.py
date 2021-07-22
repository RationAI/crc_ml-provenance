import os
import numpy as np
import sys

import polish_method


import matplotlib.pyplot as plt
import skimage.io as io

import pickle
import anhir_method as am
import utils
SAVE_TRANFORMS = True

def main(i):

    out = '/home/ubuntu/image_registration/our_method/out-2/tranformed_masks/'
    transforms_out = '/home/ubuntu/image_registration/our_method/out-2/transforms/'
    source_path = '/home/ubuntu/image_registration/our_method/out-2/images/h'+str(i) + '.png' # Source path
    target_path = '/home/ubuntu/image_registration/our_method/out-2/images/c'+str(i) + '.png' # Target path

    source = utils.load_image(source_path)
    target = utils.load_image(target_path)
    p_source, p_target, ia_source, ng_source, nr_source, i_u_x, i_u_y, u_x_nr, u_y_nr, warp_resampled_landmarks, warp_original_landmarks, return_dict = am.anhir_method(target, source)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(source, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(target, cmap='gray')
    plt.axis('off')

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(p_target, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(p_source, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(nr_source, cmap='gray')
    plt.axis('off')
    io.imsave(out+"target"+str(i) + ".png",p_target)
    io.imsave(out+"source"+str(i) + ".png", p_source)
    io.imsave(out+"transformed"+str(i) + ".png", nr_source)
    plt.show()
    if SAVE_TRANFORMS:
        pickle.dump(u_x_nr, open(transforms_out + "u_x_nr" + str(i), "wb"))
        pickle.dump(u_y_nr, open(transforms_out + "u_y_nr" + str(i), "wb"))


if __name__ == "__main__":
    for i in range(0,2):
        main(i)