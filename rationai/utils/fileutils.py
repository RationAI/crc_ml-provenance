import h5py
from pathlib import Path

from typing import Any
from typing import Union
from typing import Type


def get_h5_attr(hdf5: Union[str, Path, Type[h5py.File]], attr: str, ds_name: str = None) -> Any:
    """
    Retrieves an attribute from HDF5 and closes the file.

    Parameters
    ----------
    hdf5 : str / Path / h5py.File
        Path to a file or HDF5 file itself.

    attr : str
        Key of an HDF5 attribute whose value to retrieve.

    ds_name : str
        Name of a HDF5 Dataset whose attribute to retrieve.
        (Alternative to retrieving from attributes of an HDF5 File itself.)

    Return
    ------
    Any
        Returns contents of an HDF5 attribute stored under key `attr`.

    Raise
    -----
    ValueError
        When an attribute `attr` does not exist.

    RAI_UNTESTABLE - file handling is not tested
    """
    if type(hdf5) is not h5py.File:
        hdf5 = h5py.File(hdf5, 'r')

    file_handle = hdf5

    if ds_name is not None:
        hdf5 = hdf5[ds_name]

    if attr not in hdf5.attrs.keys():
        raise ValueError(f'Attr "{attr}" not found in {hdf5}')

    result = hdf5.attrs.get(attr)
    file_handle.close()
    return result
