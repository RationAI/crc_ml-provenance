from __future__ import annotations

import h5py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import HDF5Matrix
from typing import (
    List,
    Tuple,
    Union
)

from rationai.utils import DirStructure


class DataSource:
    """The class loads a dataset and provides a split ability.

    Sampler class is given a DataSource to build a sampling tree.

    Accepts two dataset file formats:
     - pd.DataFrame (i.e. one coordinate map)
        Split can be stratified at learning example level.

     - HDF5 file
        HDF5 dataset 'train' has to exist in the HDF5 file.
        Optionally, a predefined testing subset can be provided
        using a HDF5 dataset stored under key 'test'.
        Split is stratified if HDF5 dataset has attribute 'stratify' (array)
        that reperesents the classes.
    """

    def __init__(self,
                 dataset_config: dict,
                 dir_struct: DirStructure,
                 source: Union[List[Path], pd.DataFrame] = None):
        self.dataset_config = dataset_config
        self.dir_struct = dir_struct
        self.source = source

        self.data_dir = dir_struct.get('data_root')
        self.dataset_path = self._init_ds_path()
        self.is_composed = self.dataset_path.suffix == '.h5'

    def __len__(self) -> int:
        """Returns the length of the source"""
        return len(self.source)

    def get_train_valid_test(self) -> Tuple[DataSource, DataSource, DataSource]:
        """Splits and returns the train, validation, and test DataSource"""
        if self.is_composed:
            return self._split_composed_source()
        return self._split_single_source()

    def _init_ds_path(self) -> Path:
        """Inits dataset paths if it does not exist already"""
        ds_path = self.dir_struct.get('dataset')
        if not ds_path:
            root = self.dir_struct.get('data_root')
            ds_path = self.dir_struct.add(
                'dataset',
                root / 'datasets' / self.dataset_config['name'])
        return ds_path

    def _process_path(self, path: Path) -> Path:
        """Prefixes a path to coordinate maps with `data_root` dir."""
        path = Path(path)
        if not path.is_absolute():

            # handle the case when path contains 'data/' prefix
            if str(path).startswith('data'):
                path = str(path).replace('data/', '', 1)

            path = self.data_dir / path
        return path

    def _split_single_source(self) -> Tuple[DataSource, DataSource, DataSource]:
        """Splits a DataSource holding a pd.DataFrame dataset
        and returns a train, validation, and test DataSource.

        Config file parameters
        ----------------------
        ["data"]["dirs"]["dataset]:
            test_size: float / int
                Testing set fraction or size. (default: 0.2)

            valid_size: float / int
                Validation set fraction or size. (default: 0.1)

            stratify_col: str
                A column whose values will be used as classes for
                a stratified split.

        """
        # Load dataframe
        compression = 'gzip' if self.dataset_path.suffix == '.gz' else None
        source = pd.read_pickle(self.dataset_path, compression=compression)

        # Test split
        test_size = self.dataset_config.get('test_size', 0.2)
        stratify_col = self.dataset_config.get('stratify_col')
        stratify = source[stratify_col] if stratify_col else None
        train_source, test_source = train_test_split(source, test_size=test_size, stratify=stratify)

        # Valid split
        valid_size = self.dataset_config.get('valid_size', 0.1)
        stratify = train_source[stratify_col] if stratify_col else None
        train_source, valid_source = train_test_split(train_source, test_size=valid_size, stratify=stratify)

        return DataSource(self.dataset_config, self.dir_struct, train_source), \
            DataSource(self.dataset_config, self.dir_struct, valid_source), \
            DataSource(self.dataset_config, self.dir_struct, test_source)

    def _split_composed_source(self) -> Tuple[DataSource, DataSource, DataSource]:
        """Splits a DataSource holding an HDF5 dataset
        and returns a train, validation, and test DataSource.

        HDF5 Dataset stores paths to coordinate maps instead
        of the actual learning examples.

        Splits are stratified if HDF5 dataset contains `stratify` attribute.
        The attribute is an array with classes to be used
        in sklearn's train_test_split. The elements represent the classes
        that map to coordinate maps paths.

        Config file parameters
        ----------------------
        ["data"]["dirs"]["dataset]:
            test_size: float / int
                Testing set fraction or size. (default: 0.2)

            valid_size: float / int
                Validation set fraction or size. (default: 0.1)
        """
        # Load train dataset from .h5
        file_mode = h5py.get_config().default_file_mode
        h5py.get_config().default_file_mode = 'r'        # deprecation warn fix
        hds_train = HDF5Matrix(self.dataset_path, 'train')
        h5py.get_config().default_file_mode = file_mode  # set the mode back
        train_source = [self._process_path(p) for p in hds_train]

        # Use stratification, if exists
        stratify_list = []
        if 'stratify' in hds_train.data.attrs:
            stratify_list += hds_train.data.attrs['stratify'].tolist()

        # Try to load test dataset from .h5
        test_source = []
        try:
            hds_test = HDF5Matrix(self.dataset_path, 'test')
            test_source = [self._process_path(p) for p in hds_test]
        except Exception:
            test_size = self.dataset_config.get('test_size', 0.2)
            stratify_list = stratify_list if len(train_source) == len(stratify_list) else None
            train_source, test_source, stratify_list, _ = train_test_split(
                train_source,
                test_size=test_size,
                stratify=stratify_list)

        # Valid split
        valid_size = self.dataset_config.get('valid_size', 0.1)
        stratify_list = stratify_list if stratify_list else None
        train_source, valid_source = train_test_split(train_source,
                                                      test_size=valid_size,
                                                      stratify=stratify_list)

        return DataSource(self.dataset_config, self.dir_struct, train_source), \
            DataSource(self.dataset_config, self.dir_struct, valid_source), \
            DataSource(self.dataset_config, self.dir_struct, test_source)
