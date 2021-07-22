RESULTS_FOLDER = "results_level_0_registration_level_4_otsu_reg"

MANUAL_CROP_X_BOUNDS = {'lower': 6000 * 16, 'upper': 12000 * 16}

# ************ segment samples constants *****
SEGMENTATION_LEVEL = 7
PROCESSING_LEVEL = 1

SEGMENTATION_WHITE_SAT_LOWER_THR = 0.05
SEGMENTATION_WHITE_SAT_UPPER_THR = 1.0
SEGMENTATION_RUBBISH_HUE_THR = 0.4

SEGMENTATION_CLOSING_DIAM_HE = 5
SEGMENTATION_CLOSING_DIAM_CE = 5

# Minimum object size in the input image
MIN_SAMPLE_AREA = 100

# ************** Color segmentation ***********

COLOR_SEG_KERNEL_SIZE = 3
COLOR_SEG_MAX_DIST = 6
COLOR_SEG_RATIO = 0.5

# ************** segment nuclei constants ************

HE_NUCLEI_MAX_AREA = 300
HE_NUCLEI_MIN_AREA = 50

CE_NUCLEI_MAX_AREA = 250
CE_NUCLEI_MIN_AREA = 30

CYTOKERATIN_OPTIMAL_NUMBER_NUCLEI = 300
HEMATOXYLIN_OPTIMAL_NUMBER_NUCLEI = 4000

NUCLEI_SEG_COLOR_THR_MIN = 0.1
NUCLEI_SEG_COLOR_THR_MAX = 0.8
NUCLEI_SEG_COLOR_THR_STEPS = 100

# ************** fill holes ******

# Minimum area of the hole to be filled
HOLES_MIN_AREA = 50
# Maximum area ...
HOLES_MAX_AREA = 200
# The lower bound on the proportion of the HE stained subarea of a hole to be filled
HOLES_H_PROPORTION = 0.4

# ************ Optimization constants **************

# ************ Hierarchical grid search ************

NUMBER_OF_PARALLEL_GRIDS = 60
NUMBER_OF_STEPS_GRID_SEARCH_EXP = 4
TOP_STEP_SIZE_GRID_EXP = 7
BOT_STEP_SIZE_GRID_EXP = 0

STOPPING_BAD_SUFFIX_LENGTH = 5

# ************ Search for angle ***************

ANGLE_STEP = 0.01
NUMBER_OF_ANGLE_STEPS = 200

# ************ LOCAL search ********************

LOCAL_SEARCH_NUMBER_OF_STEPS = 4
LOCAL_SEARCH_STEP_SIZE = 2