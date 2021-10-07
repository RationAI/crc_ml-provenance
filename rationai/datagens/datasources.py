# Standard Imports
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from typing import Union
from typing import Tuple
from typing import Optional
from enum import Enum

# Third-party Imports
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.io.pytables import HDFStore


# Local Imports


class DataSource(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self):
        raise NotImplementedError

    @abstractmethod
    def split(self):
        raise NotImplementedError


class HDF5_DataSource(DataSource):
    def __init__(self, dataset_fp: Path, tables: List[str], source: HDFStore):
        self.dataset_fp = dataset_fp
        self.tables = tables
        self.source = source

    def get_table(self, table_key):
        return self.source.select(table_key)

    def get_metadata(self, key: str) -> Optional[dict]:
        try:
            return self.source.get_storer(key).attrs.metadata
        except AttributeError:
            return {}

    def load_dataset(dataset_fp: Path, keys: List[str]) -> HDF5_DataSource:
        """Loads the dataset as a union of all tables across specified keys.

        Args:
            dataset_fp (Path): Path to the HDF5 dataset.
            keys (List[str]): Keys that should be included in the result set.

        Returns:
            HDF5_DataSource: DataSource
        """
        source = pd.HDFStore(dataset_fp, 'r')

        tables = []
        for key in keys:
            tables += [node._v_pathname for node in source.get_node(str(key))]

        return HDF5_DataSource(dataset_fp, tables=tables, source=source)

    def split(self, sizes: List[float], key: Optional[str]) -> List[HDF5_DataSource]:
        """Partition the DataSource into N partitions. The size of each partition is defined by
        `sizes` parameter. Key defines how the DataSource is split.

        Args:
            sizes (List[float]): Defines size of new DataSource as a fraction of the old one.
            key (Optional[str]): When `None` the DataSource is split on the HDF5 key of each table.
                If specified, the value of metadata attribute key is used.

        Returns:
            List[HDF5_DataSource]: List of DataSource partitions.
        """
        data_sources = []
        tables = self.tables
        n_tables = len(self.tables)
        for size in sizes[:-1]:
            if key is None:
                stratify = [table_key.rsplit('/', 1)[0] for table_key in tables]
            else:
                stratify = [
                    self.source.get_storer(table_key).attrs[key] for table_key in tables
                ]
            new_tables, tables = train_test_split(
                tables,
                train_size=int(n_tables*size),
                stratify=stratify
            )
            data_sources.append(
                HDF5_DataSource(
                    self.dataset_fp,
                    new_tables,
                    self.source
                )
            )
        data_sources.append(
                HDF5_DataSource(
                    self.dataset_fp,
                    tables,
                    self.source
                )
            )
        return data_sources