import numpy as np
import tqdm
from skimage.color import rgba2rgb
from rationai.data.imreg.muni.sample_segmentation.segment_samples_tools import get_segments
from rationai.data.imreg.muni.sample_segmentation.segment_samples_tools import get_ordered_segments
from rationai.data.imreg.muni.sample_segmentation.segment_samples_tools import get_segment_from_openslide
from rationai.data.imreg.muni.sample_segmentation.segment_samples_tools import pad_bounding_box
from rationai.data.imreg.muni.utils.image_tools import to_shape_white_padding


def get_samples_generator(he_openslide, ce_openslide, level, he_ignore_annotation, ce_ignore_annotation,
                          select_samples, blur_radius, white_thresh, minimum_area, segmentation_level,
                          min_sample_area):

    print('Segmenting HE and DAB slides ... ', end='')
    segments_he = get_segments(he_openslide,
                               segmentation_level,
                               min_sample_area,
                               he_ignore_annotation,
                               blur_radius,
                               white_thresh,
                               minimum_area
                               )

    segments_ce = get_segments(ce_openslide,
                               segmentation_level,
                               min_sample_area,
                               ce_ignore_annotation,
                               blur_radius,
                               white_thresh,
                               minimum_area
                               )
    print('DONE')

    segments_he, segments_ce = get_ordered_segments(segments_he, segments_ce)

    segments_he = np.asarray(list(map(lambda x: pad_bounding_box(x, 5), segments_he)))
    segments_ce = np.asarray(list(map(lambda x: pad_bounding_box(x, 5), segments_ce)))

    if len(select_samples) > 0:
        segments_he = [s for i, s in enumerate(segments_he) if i in select_samples]
        segments_ce = [s for i, s in enumerate(segments_ce) if i in select_samples]

    print(f'Segmentation recognized {len(segments_he)} tissue cores')
    print('Extracting segments from WSIs')
    for segment_he, segment_ce in tqdm.tqdm(zip(segments_he, segments_ce),
                                            total=len(segments_he)):
        he_sample, he_annotation, he_smaller = get_segment_from_openslide(he_openslide, segment_he,
                                                                          segmentation_level, level,
                                                                          he_ignore_annotation)

        ce_sample, ce_annotation, ce_smaller = get_segment_from_openslide(ce_openslide, segment_ce,
                                                                          segmentation_level, level,
                                                                          ce_ignore_annotation)

        he_sample = he_sample[:, :, :4] / 255
        ce_sample = ce_sample[:, :, :4] / 255
        he_sample = rgba2rgb(he_sample)
        ce_sample = rgba2rgb(ce_sample)
        max_shape = np.maximum(he_sample.shape, ce_sample.shape)

        yield {
            "he_sample": to_shape_white_padding(he_sample, max_shape),
            "ce_sample": to_shape_white_padding(ce_sample, max_shape),
            "he_annotation": he_annotation,
            "ce_annotation": ce_annotation,
            "he_smaller": rgba2rgb(he_smaller),
            "ce_smaller": rgba2rgb(ce_smaller)
        }
