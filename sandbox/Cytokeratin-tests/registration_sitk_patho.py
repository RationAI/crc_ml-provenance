#!/usr/bin/python3

import skimage
from skimage import data, io, filters
from skimage.color import rgb2gray, gray2rgb

import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed
import skimage
import skimage.morphology
import skimage.exposure
import skimage.util
import imageio
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import numpy as np

from skimage.filters import threshold_otsu

from skimage.morphology import diamond, binary_dilation, closing, square

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

import math

import SimpleITK as sitk

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 30000000000


def rgb2h(im):
    im_rgb = im[:,:,:3]
    im_hed = rgb2hed(im_rgb)
    im_h = im_hed[:, :, 0]
    return skimage.exposure.rescale_intensity(im_h, out_range=(0,1))

#The crop is not the part of the algorithm
im_ce = io.imread("CK-DAB-H-HR2-16PgR-B.png")[6000:12000, ...]
im_he = io.imread("HE-HR2-16PgR-B.png")[6000:12000, ...]

im_ce_gray = rgb2gray(im_ce)
im_he_gray = rgb2gray(im_he)

#im_ce_h = skimage.exposure.rescale_intensity(rgb2h(im_ce), out_range=(0,255))
#im_he_h = skimage.exposure.rescale_intensity(rgb2h(im_he), out_range=(0,255))
im_ce_h = rgb2h(im_ce)
im_he_h = rgb2h(im_he)

# this is just to get basics working - using SimpleElastix demo
# simple rigid transformation - translation + rotation
elastixImageFilter = sitk.ElastixImageFilter()
#im_fixed = sitk.GetImageFromArray(im_he_h.astype(np.uint8))
im_fixed = sitk.GetImageFromArray(im_he_h.astype(np.float32))
elastixImageFilter.SetFixedImage(im_fixed);
#im_moving = sitk.GetImageFromArray(im_ce_h.astype(np.uint8))
im_moving = sitk.GetImageFromArray(im_ce_h.astype(np.float32))
elastixImageFilter.SetMovingImage(im_moving);
#elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("translation"))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
#elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap("bspline"))
bsplineMap = sitk.GetDefaultParameterMap("bspline")
#bsplineMap['GridSpacingSchedule'] = ['2.803220', '1.988100', '1.410000', '1.000000']
#bsplineMap['GridSpacingSchedule'] = ['2.803220', '1.988100', '1.410000', '256.000000']
bsplineMap['GridSpacingSchedule'] = ['5.57308', '2.803220', '1.410000', '256.000000']
#bsplineMap['NumberOfResolutions'] = ['6.000000']
#bsplineMap['GridSpacingSchedule'] = ['22.0278451', '11.0798476', '5.57308', '2.803220', '1.410000', '256.000000']
#bsplineMap['GridSpacingSchedule'] = ['5.57308', '3.95254', '2.80322', '1.9881', '1.410000', '128.000000']
elastixImageFilter.AddParameterMap(bsplineMap)
elastixImageFilter.Execute()
sitk.WriteImage(im_fixed, "registrationResult-CK-fixed.tif")
sitk.WriteImage(im_moving, "registrationResult-CK-moving.tif")
im_result = elastixImageFilter.GetResultImage()
sitk.WriteImage(im_result, "registrationResult-CK-result.tif")
im_composed = np.dstack((sitk.GetArrayFromImage(im_fixed),sitk.GetArrayFromImage(im_moving),sitk.GetArrayFromImage(im_result)))
im_composed_source = np.dstack((sitk.GetArrayFromImage(im_fixed),sitk.GetArrayFromImage(im_moving),np.zeros((im_fixed.GetHeight(),im_fixed.GetWidth()),dtype=np.uint8)))
im_composed_result = np.dstack((sitk.GetArrayFromImage(im_fixed),sitk.GetArrayFromImage(im_result),np.zeros((im_fixed.GetHeight(),im_fixed.GetWidth()),dtype=np.uint8)))
#im_composed = np.dstack((sitk.GetArrayFromImage(im_fixed),sitk.GetArrayFromImage(im_moving),np.zeros((im_fixed.GetHeight(),im_fixed.GetWidth()),dtype=np.uint8)))
io.imsave("registrationResult-CK-result-composed.tif", im_composed)
io.imsave("registrationResult-CK-result-composed-source.tif", im_composed_source)
io.imsave("registrationResult-CK-result-composed-result.tif", im_composed_result)


