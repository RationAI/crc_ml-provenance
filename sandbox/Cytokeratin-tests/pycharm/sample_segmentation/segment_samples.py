import numpy as np

from magic_constants import SEGMENTATION_WHITE_SAT_LOWER_THR, SEGMENTATION_WHITE_SAT_UPPER_THR, \
    SEGMENTATION_RUBBISH_HUE_THR, MIN_SAMPLE_AREA, SEGMENTATION_LEVEL, \
    SEGMENTATION_CLOSING_DIAM_CE, SEGMENTATION_CLOSING_DIAM_HE
from sample_segmentation.segment_samples_tools import get_segments, get_ordered_segments, get_segment_from_openslide
from utils.image_tools import to_shape


def get_samples_generator(he_openslide, ce_openslide, level, he_pre_crop, ce_pre_crop, select_samples=[]):
    segments_he, mask_he = get_segments(he_openslide,
                                        SEGMENTATION_LEVEL,
                                        MIN_SAMPLE_AREA,
                                        SEGMENTATION_WHITE_SAT_LOWER_THR,
                                        SEGMENTATION_WHITE_SAT_UPPER_THR,
                                        SEGMENTATION_RUBBISH_HUE_THR,
                                        SEGMENTATION_CLOSING_DIAM_HE,
                                        he_pre_crop
                                        )

    segments_ce, mask_ce = get_segments(ce_openslide,
                                        SEGMENTATION_LEVEL,
                                        MIN_SAMPLE_AREA,
                                        SEGMENTATION_WHITE_SAT_LOWER_THR,
                                        SEGMENTATION_WHITE_SAT_UPPER_THR,
                                        SEGMENTATION_RUBBISH_HUE_THR,
                                        SEGMENTATION_CLOSING_DIAM_CE,
                                        ce_pre_crop
                                        )

    segments_he, segments_ce = get_ordered_segments(segments_he, segments_ce)

    if len(select_samples) > 0:
        segments_he = [s for i, s in enumerate(segments_he) if i in select_samples]
        segments_ce = [s for i, s in enumerate(segments_ce) if i in select_samples]

    print("Analyze ", len(segments_he), " samples. ")
    print("Selected sample numbers ", select_samples)

    for segment_he, segment_ce in zip(segments_he, segments_ce):
        he_sample = get_segment_from_openslide(he_openslide, segment_he,
                                               SEGMENTATION_LEVEL, level)[:, :, :3] / 255
        ce_sample = get_segment_from_openslide(ce_openslide, segment_ce,
                                               SEGMENTATION_LEVEL, level)[:, :, :3] / 255

        max_shape = np.maximum(he_sample.shape, ce_sample.shape)

        yield {
            "he_sample": to_shape(he_sample, max_shape),
            "ce_sample": to_shape(ce_sample, max_shape)
        }
