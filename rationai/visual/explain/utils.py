"""
General utility functions which don't have a home anywhere else.

This module should be periodically checked for entities that should be kept
elsewhere.
"""
# Standard imports
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Union

# Third-party imports
import cv2
import pandas as pd


def make_directory_structure(path):
    """
    Create a directory structure.

    Do so only if such a structure doesn't exist yet and doesn't conflict with
    an existing path.

    Parameters
    ----------
    path : str
        Desired directory structure to create.
    """
    full_path = ''

    for directory in path.split('/'):
        full_path += directory + '/'
        if not os.path.exists(full_path):
            os.mkdir(full_path)


def extract_coordinates(coord_file_path: Union[Path, str]):
    """
    Extract tile coordinates from a saved pandas dataframe pickle.

    Note that the semantics may change depending on the file. Currently, it is
    expected that each row corresponds to a middle tile of a macrotile, and the
    coordinates are given in level=0 resolution.

    Parameters
    ----------
    coord_file_path : str
        Path to a file with a saved pandas dataframe pickle. Each row must
        contain coord_x and coord_y values.

    Returns
    -------
    list(tuple(int, int))
        X and Y coordinates of macrotiles.
    """
    coordinate_tuples = []
    for item in pd.read_pickle(Path(coord_file_path)).iterrows():
        coordinate_tuples.append((int(item[1].coord_x), int(item[1].coord_y)))
    return coordinate_tuples


def get_map_name(image_idx, coord_x, coord_y, suffix='.npy'):
    """
    TODO: Add docs
    """
    return str(image_idx) + '-' + str(int(coord_x)) + '-' + str(int(coord_y)) + suffix


def get_interpolation_strategy(interpolation_name):
    """
    Resolve cv2 interpolation strategy flag from string id.

    Parameters
    ----------
    interpolation_name : str
        ID of the interpolation strategy.

    Returns
    -------
    int
        A cv2 interpolation strategy flag.
    """
    if interpolation_name == 'nearest':
        return cv2.INTER_NEAREST
    if interpolation_name == 'linear':
        return cv2.INTER_LINEAR
    if interpolation_name == 'area':
        return cv2.INTER_AREA
    if interpolation_name == 'cubic':
        return cv2.INTER_CUBIC
    if interpolation_name == 'lancos4':
        return cv2.INTER_LANCZOS4

    raise ValueError('Unknown interpolation type')


def copy_config_json(output_dir, config_to_copy, additional_info=None):
    """
    Copy config file and add additional info - parameters with which the script
    was called.

    Parameters
    ----------
    output_dir : str
        Output directory path.
    config_to_copy : str
        Path to the configuration file to be copied.
    additional_info : dict
        A dictionary of keyvalue pairs to be added to the config copy.
    """
    out_file_path = os.path.join(output_dir, 'config.json')
    copyfile(config_to_copy, out_file_path)
    with open(out_file_path, 'r') as infile:
        json_content = json.load(infile)

    if additional_info is not None:
        for key, value in additional_info.items():
            json_content[key] = value

    with open(out_file_path, 'w') as outfile:
        json.dump(json_content, outfile)
