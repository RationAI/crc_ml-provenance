import numpy as np
import matplotlib.pyplot as plt
import cv2

# first cat image
C1 = cv2.imread('cat1.jpg')
C1_gr = cv2.cvtColor(C1, cv2.COLOR_BGR2GRAY)

# second cat image
C2 = cv2.imread('cat2.jpg')
C2_gr = cv2.cvtColor(C2, cv2.COLOR_BGR2GRAY)

# hand-pick a keypoint
kp1 = (719, 769)
kp2 = (903, 728)

# show images
marker_style = {
    'markersize': 15,
    'markeredgewidth': 3,
    'markeredgecolor': 'w',
    'markerfacecolor': 'None',
    }

# 8 random points in "cat1.jpg"
rnd_pts_1 = np.array(
    [[ 266, 1147],
     [ 896,  884],
     [ 385,  566],
     [ 468,  141],
     [ 889, 1084],
     [ 549, 1029],
     [ 987,  145],
     [ 419,  931]])

# 8 random points in "cat2.jpg"
rnd_pts_2 = np.array(
    [[ 811,  980],
     [1176,  716],
     [ 259,  340],
     [ 745,  952],
     [ 265,  730],
     [ 852,  774],
     [1019, 1127],
     [ 660, 1110]])

# show images
f, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(C1_gr, cmap='gray', interpolation='none')
ax[0].set_title('cat 1')
ax[1].imshow(C2_gr, cmap='gray', interpolation='none')
ax[1].set_title('cat 2')
for i in range(rnd_pts_1.shape[0]):
    label = 'rand-' + str(i+1)
    _marker_style = dict(marker_style)
    _marker_style['markeredgecolor'] = 'C' + str(i if i < 7 else i+1)
    ax[0].plot(*rnd_pts_1[i], 'o', label=label, **_marker_style)
    ax[1].plot(*rnd_pts_2[i], 'o', label=label, **_marker_style)
ax[1].legend(
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.0,
    labelspacing=1,
    borderpad=1)

def get_bin_desc(extractor, img, kp, size=30):
    # extract a binary descriptor from the image
    d = extractor.compute(img, [cv2.KeyPoint(*kp, size)])[1][0]
    return np.unpackbits(d)

def get_hamming_dist(d1, d2):
    # return the Hamming distance
    return np.sum(np.logical_xor(d1, d2))

def run_bin_test(extractor):
    # print Hamming distances between keypoints using the binary extractor

    d1 = get_bin_desc(extractor, C1_gr, kp1)
    d2 = get_bin_desc(extractor, C2_gr, kp2)
    dist = get_hamming_dist(d1, d2)

    print('Hamming distance of pairs (out of {})\n'.format(len(d1)))
    print('Hand-picked: {:3d}'.format(dist))

    for i in range(rnd_pts_1.shape[0]):
        d1 = get_bin_desc(extractor, C1_gr, rnd_pts_1[i])
        d2 = get_bin_desc(extractor, C2_gr, rnd_pts_2[i])
        dist = get_hamming_dist(d1, d2)

        print('Random {}:    {:3d}'.format(i+1, dist))

    print("")

#extractor = cv2.xfeatures2d_FREAK.create()
#print("FREAK results")
#run_bin_test(extractor)
#
#extractor = cv2.ORB.create()
#print("ORB results")
#run_bin_test(extractor)

extractor = cv2.BRISK.create()
print("BRISK results")
run_bin_test(extractor)

#extractor = cv2.xfeatures2d_BriefDescriptorExtractor.create()
#print("BRIEF results")
#run_bin_test(extractor)

# these are patented
#extractor = cv2.xfeatures2d_SIFT.create()
#print("SIFT results")
#run_bin_test(extractor)
#
#extractor = cv2.xfeatures2d_SURF.create()
#print("SURF results")
#run_bin_test(extractor)

#brisk = cv2.BRISK.create(octaves = 3, patternScale = 10.0)
brisk = cv2.BRISK.create(octaves = 6, patternScale = 3.0)
kp1, des1 = brisk.detectAndCompute(C1_gr,None)
kp2, des2 = brisk.detectAndCompute(C2_gr,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_CENTERS_RANDOM = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# this is because BRISK does not produce float descriptors hamming distance needs to be used instead of euclidian distance
index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
#index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 12, key_size = 20, multi_probe_level = 2)
#index_params = dict(algorithm = FLANN_INDEX_HIERARCHICAL, branching = 32, centers_init = FLANN_CENTERS_RANDOM, trees = 4, leaf_size = 100)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(C1,kp1,C2,kp2,matches,None,**draw_params)
cv2.imwrite("cat-cmp.jpg",img3)
