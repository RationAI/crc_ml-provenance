import math
import numpy as np
import skimage
from concurrent.futures import ProcessPoolExecutor

# TODO: Check if it still works after *args **kwargs chaining
def _process_args_parallel(pool, func, args_list, **kvargs):
    futures = []
    results = []

    if pool is None:
        pool = ProcessPoolExecutor()

    for block in args_list:
        if len(kvargs.keys()) > 0:
            futures.append(pool.submit(func, *block, **kvargs))
        else:
            futures.append(pool.submit(func, *block))

    for f in futures:
        results.append(f.result())

    return results


# func must accept (i, j, block) as an argument
def _apply_to_blocks(img_cropped, func, block_size, blocks_number, *args, **kwargs):
    if len(img_cropped.shape) == 3:
        blocks = skimage.util.view_as_blocks(img_cropped, block_shape=(block_size[0], block_size[1], img_cropped.shape[2]))
        blocks_list = [
            (i, j, blocks[i, j, 0, : :, :])
            for i in range(blocks_number[0])
            for j in range(blocks_number[1])]
    else:
        blocks = skimage.util.view_as_blocks(img_cropped,
                                             block_shape=(block_size[0], block_size[1]))
        blocks_list = [
            (i, j, blocks[i, j, :, :])
            for i in range(blocks_number[0])
            for j in range(blocks_number[1])]

    return _process_args_parallel(None, func, blocks_list, *args, **kwargs), blocks_list


def _get_block_size_and_img_cropped(img, blocks_number):
    block_size = (
        math.floor(img.shape[0] / blocks_number[0]),
        math.floor(img.shape[1] / blocks_number[1]),
    )

    img_cropped = img[0:block_size[0] * blocks_number[0], 0:block_size[1] * blocks_number[1]]

    return block_size, img_cropped


def process_image_by_blocks(img, func, blocks_number, *args, **kwargs):
    block_size, img_cropped = _get_block_size_and_img_cropped(img, blocks_number)
    processed_blocks, _ = _apply_to_blocks(img_cropped, func, block_size, blocks_number, *args, **kwargs)

    result = np.zeros(img_cropped.shape[:len(processed_blocks[0][2].shape)])

    for pblock in processed_blocks:
        i = pblock[0]
        j = pblock[1]
        block = pblock[2]

        result[
        i * block.shape[0]:(i + 1) * block.shape[0],
        j * block.shape[1]:(j + 1) * block.shape[1]
        ] += block

    return result


def compute_image_features_by_blocks(img, func, blocks_number):
    block_size, img_cropped = _get_block_size_and_img_cropped(img, blocks_number)
    processed_blocks = _apply_to_blocks(img_cropped, func, block_size, blocks_number)
    return processed_blocks
