# Standard Imports
from __future__ import annotations
from pathlib import Path
from typing import List
from typing import Union
from typing import Tuple
from typing import Any

# Third-party Imports
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py


# Local Imports

class DataSource:
    def __init__(self, dataset_fp: Path, source: List):
        self.dataset_fp = dataset_fp
        self.source = source

    def __len__(self):
        return len(self.source)

    def loadDataset(dataset_fp: Path, *args: Any, **kwargs: Any) -> List[DataSource]:
        if dataset_fp.suffix == '.gz':
            return HDF5_DataSource.loadDataset(dataset_fp, *args, **kwargs)
        return ValueError(f'Unknown file format: {dataset_fp.suffix}.')

    def split(self):
        raise NotImplementedError


class HDF5_DataSource(DataSource):
    def loadDataset(dataset_fp: Path, keys: List[str]) -> List[HDF5_DataSource]:
        dataset_hdf5 = h5py.File(dataset_fp, 'r')
        datasources = []
        for key in keys:
            datasources.append(HDF5_DataSource(dataset_fp, dataset_hdf5[key]))
        return datasources

    def split(self, new_size: Union[int, float],
              stratify: bool) -> Tuple[HDF5_DataSource, HDF5_DataSource]:
        dataset_hdf5 = h5py.File(self.dataset_fp, 'r')

        stratify_list = None
        if stratify and ('stratify' in dataset_hdf5.data.attrs):
            stratify_list = dataset_hdf5.data.attrs['stratify'].tolist()

        src1, src2 = train_test_split(self.source, test_size=new_size, stratify=stratify_list)
        return HDF5_DataSource(self.dataset_fp, src1), HDF5_DataSource(self.dataset_fp, src2)