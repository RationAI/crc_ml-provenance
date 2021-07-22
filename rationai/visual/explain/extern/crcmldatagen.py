"""
Data generators for various CRCML data sources.
"""
# Standard imports
from pathlib import Path
from nptyping import NDArray
from typing import Generator

# Project-specific imports (external)
from rationai.datagens import Datagen
from rationai.datagens import DataSource
from rationai.utils import DirStructure


def get_datasource_for_test_slide_tf2(
        params: dict,
        coord_map_path: Path,
        dir_struct: DirStructure) -> Generator[NDArray, None, None]:
    """
    Returns batches of tiles from a test slide.

    For use with TensorFlow 2+ (VGGPRETRAINED) tile models.

    The returned matrix is of shape:
        (batch_size, num_tiles, tile_size, tile_size, 3)
    where 3 stands for color channels.

    Parameters
    ----------
    params : dict
        Main config file.
    coord_map_path : Path
        Path to the slide's coord_map file.
    dir_struct : DirStructure
        Path manager.
    Yields
    ------
    array-like
        Batches of tiles for a given test slide.
    """
    datasource = DataSource(params['data']['dirs']['dataset'],
                            dir_struct=dir_struct,
                            source=[coord_map_path])

    datagen = Datagen(params, dir_struct=dir_struct)

    test_gen = datagen.get_sequential_generator(datasource, augment_type=False)

    for batch in test_gen:
        # always batch_size = 1 -> get the first element of the batch
        yield batch[0]
