"""
Joining of occlusion images into a map covering the whole slide image.
"""
# Standard imports
import os

# Third-party imports
import cv2
import numpy as np
import openslide as oslide
from pathlib import Path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scale(x, old_min, old_max, new_min=0, new_max=1):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def occlusion_to_image(occ_map, img_height, img_width, interpolation=cv2.INTER_LINEAR, border=True):
    """
    Converts an occlusion map to a more interpretable image with the same
    dimensions as the original region.

    Parameters
    ----------
    occ_map : array-like
        The occlusion map.
    img_height : int
        Height of the original image in pixels.
    img_width : int
        Width of the original image in pixels.
    interpolation : int
        OpenCV's interpolation strategy flag.
    border : bool
        Whether to outline the explanation of the given region with a white
        border.

    Returns
    -------
    array-like
        A more interpretable image, with the same dimensions as the original
        region.
    """
    # Get original image dimension.
    image = np.zeros((img_height, img_width, 4), dtype=np.uint16)
    heatmap = cv2.resize(
        occ_map, (img_height, img_width), interpolation=interpolation
    ).astype(np.float32)

    heatmap_scaled = scale(sigmoid(heatmap) - .5, -.5, .5, -1, 1)
    for x_idx, row in enumerate(heatmap_scaled):
        for y_idx, col in enumerate(row):
            value = heatmap_scaled[x_idx, y_idx]
            if value > 0:
                image[x_idx, y_idx, 1] = int(255 * value)
            elif value < 0:
                image[x_idx, y_idx, 0] = int(-255 * value)

    if border:
        image[0, :, :] = 255
        image[-1, :, :] = 255
        image[:, 0, :] = 255
        image[:, -1, :] = 255
    return image


def extract_coords_file_1_96_3(filename):
    """
    Extract coordinates from a standard macrotile explanation filename.
    """
    return (
        (int(filename.split('-')[1]) - 192) // 2,
        (int(filename.split('-')[2].rstrip('.npy')) - 192) // 2
    )


def extract_coords_file_1_512_1(filename):
    """
    Extract coordinates from a standard single tile explanation filename.
    """
    return (
        (int(filename.split('-')[1])) // 2,
        (int(filename.split('-')[2].rstrip('.npy'))) // 2
    )


def get_occlusion_and_counts_canvas(
        slide_path: Path,
        heatmaps_dir,
        heatmap_filenames,
        img_height,
        img_width,
        coord_extractor,
        interpolation=cv2.INTER_LINEAR,
        border=True
):
    """
    Joins given occlusion heatmaps into a single annotation layer over the
    original WSI image. Overlays are saved as sum and count numpy memmaps on
    disk.

    Parameters
    ----------
    slide_path : str or Path-like
        Path to the original WSI image.
    heatmaps_dir : str or Path-like
        Path to the directory with occlusion heatmaps.
    heatmap_filenames : list(str)
        Filenames of the occlusion heatmaps in `heatmaps_dir`.
    img_height : int
        Height of the original images from which the occlusion heatmaps were
        generated.
    img_width : int
        Width of the original images from which the occlusion heatmaps were
        generated.
    coord_extractor : callable
        Function to extract appropriate coordinates from the filename of a raw
        explanation map.
    interpolation : int
        OpenCV's interpolation strategy flag.
    border : boolean
        Whether to outline the explanation of a single input region with a
        white border.
    """
    open_slide = oslide.open_slide(str(slide_path.resolve()))
    sum_canvas_shape = (
        open_slide.level_dimensions[1][1],
        open_slide.level_dimensions[1][0],
        4
    )
    count_canvas_shape = (
        open_slide.level_dimensions[1][1],
        open_slide.level_dimensions[1][0],
        1
    )

    canvas = np.memmap(
        os.path.join(heatmaps_dir, 'sum-canvas.npy'),
        dtype=np.uint16,
        mode='w+',
        shape=sum_canvas_shape
    )
    count_canvas = np.memmap(
        os.path.join(heatmaps_dir, 'count-canvas.npy'),
        dtype=np.uint16,
        mode='w+',
        shape=count_canvas_shape
    )
    for idx, filename in enumerate(heatmap_filenames):
        occ_map = np.load(
            os.path.join(heatmaps_dir, filename), allow_pickle=True
        )
        coords = coord_extractor(filename)
        occ_image = occlusion_to_image(
            occ_map, img_height, img_width, interpolation, border
        )
        try:
            canvas[
                coords[1]:coords[1] + img_height,
                coords[0]:coords[0] + img_width
            ] += occ_image
            count_canvas[
                coords[1]:coords[1] + img_height,
                coords[0]:coords[0] + img_width
            ] += 1
        except ValueError:
            new_shape = canvas[
                coords[1]:coords[1] + img_height,
                coords[0]:coords[0] + img_width
            ].shape
            print("clipping from", occ_image.shape, "to", new_shape)
            canvas[
                coords[1]:coords[1] + img_height,
                coords[0]:coords[0] + img_width
            ] += occ_image[:new_shape[0], :new_shape[1]]
            count_canvas[
                coords[1]:coords[1] + img_height,
                coords[0]:coords[0] + img_width
            ] += 1

        if idx % 50 == 0:
            del canvas
            del count_canvas
            canvas = np.memmap(
                os.path.join(heatmaps_dir, 'sum-canvas.npy'),
                dtype=np.uint16,
                mode='r+',
                shape=sum_canvas_shape
            )
            count_canvas = np.memmap(
                os.path.join(heatmaps_dir, 'count-canvas.npy'),
                dtype=np.uint16,
                mode='r+',
                shape=count_canvas_shape
            )

    del canvas
    del count_canvas


def create_canvas(canvas_dir, slide_path: Path):
    """
    Create the resulting canvas from `sum-canvas.npy` and `count-canvas.npy`

    The result is saved in the directory in which the input canvases are
    situated, as `result-canvas.npy`.

    Parameters
    ----------
    canvas_dir : path-like
        The directory containing the two intermediate canvases.
    slide_path : path-like
        The path containing the MIRAX slide for which the canvas overlay
        is being generated
    """
    open_slide = oslide.open_slide(str(slide_path.resolve()))
    sum_canvas_shape = (
        open_slide.level_dimensions[1][1], open_slide.level_dimensions[1][0], 4
    )
    count_canvas_shape = (
        open_slide.level_dimensions[1][1], open_slide.level_dimensions[1][0], 1
    )

    canvas = np.memmap(
        os.path.join(canvas_dir, 'sum-canvas.npy'),
        dtype=np.uint16,
        mode='r',
        shape=sum_canvas_shape
    )
    count_canvas = np.memmap(
        os.path.join(canvas_dir, 'count-canvas.npy'),
        dtype=np.uint16,
        mode='r',
        shape=count_canvas_shape
    )
    result_canvas = np.memmap(
        os.path.join(canvas_dir, 'result-canvas.npy'),
        dtype=np.uint8,
        mode='w+',
        shape=sum_canvas_shape
    )

    for x in range(0, canvas.shape[0] + 1, 500):
        if (x // 500) % 20 == 0:
            del canvas
            del count_canvas
            del result_canvas
            canvas = np.memmap(
                os.path.join(canvas_dir, 'sum-canvas.npy'),
                dtype=np.uint16,
                mode='r',
                shape=sum_canvas_shape
            )
            count_canvas = np.memmap(
                os.path.join(canvas_dir, 'count-canvas.npy'),
                dtype=np.uint16,
                mode='r',
                shape=count_canvas_shape
            )
            result_canvas = np.memmap(
                os.path.join(canvas_dir, 'result-canvas.npy'),
                dtype=np.uint8,
                mode='r+',
                shape=sum_canvas_shape
            )
        for y in range(0, canvas.shape[1] + 1, 500):
            sub_sumcanvas = (canvas[x:x + 500, y:y + 500]).astype(np.int32)
            sub_countcanvas = (count_canvas[x:x + 500, y:y + 500]).astype(np.int32)
            differences = sub_sumcanvas[:, :, 1] - sub_sumcanvas[:, :, 0]

            red_values = (-differences[np.where(differences < 0)]) // sub_countcanvas[np.where(differences < 0)][:, 0]
            red_values = np.expand_dims(red_values, -1)
            red_zeros = np.zeros(red_values.shape)
            red_values = np.concatenate((red_values, red_zeros, red_zeros, red_zeros), axis=-1)

            green_values = differences[np.where(differences > 0)] // sub_countcanvas[np.where(differences > 0)][:, 0]
            green_values = np.expand_dims(green_values, -1)
            green_zeros = np.zeros(green_values.shape)
            green_values = np.concatenate((green_zeros, green_values, green_zeros, green_zeros), axis=-1)

            result_canvas[x:x + 500, y:y + 500][np.where(differences < 0)] = red_values.astype(np.uint8)
            result_canvas[x:x + 500, y:y + 500][np.where(differences > 0)] = green_values.astype(np.uint8)
            result_canvas[x:x + 500, y:y + 500][:, :, 3] = 255

    del canvas
    del count_canvas
    del result_canvas
