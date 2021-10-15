"""
TODO: Missing docstring.
"""
# Standard Imports
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from typing import Optional

# Third-party Imports
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.io.pytables import HDFStore


# Local Imports


class DataSource(ABC):
    """
    TODO: Missing docstring.
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    # FIXME: If this is supposed to be a static or class method, self should not be present, but proper annotation
    #        should.
    # TODO: Should this accept parameters? If so, they should be defined here.
    def load_dataset(self):
        """
        TODO: Missing docstring.
        """
        raise NotImplementedError

    @abstractmethod
    # TODO: Should this accept parameters? If so, they should be defined here.
    def get_metadata(self):
        """
        TODO: Missing docstring.
        """
        raise NotImplementedError

    @abstractmethod
    # TODO: Should this accept parameters? If so, they should be defined here.
    def split(self):
        """
        TODO: Missing docstring.
        """
        raise NotImplementedError


class HDF5DataSource(DataSource):
    """
    TODO: Missing docstring.
    """
    def __init__(self, dataset_fp: Path, tables: List[str], source: HDFStore):
        self.dataset_fp = dataset_fp
        self.tables = tables
        self.source = source

    def get_table(self, table_key):
        """
        TODO: Missing docstring.
        TODO: This should be reworked.
        """
        return self.source.select(table_key)

    def get_metadata(self, key: str) -> Optional[dict]:
        """
        TODO: Missing docstring.
        TODO: This should be reworked.
        """
        try:
            return self.source.get_storer(key).attrs.metadata
        except AttributeError:
            return {}

    # FIXME: This is currently not defined as a classmethod, so it is missing 'self'.
    def load_dataset(dataset_fp: Path, keys: List[str]) -> HDF5DataSource:
        """Loads the dataset as a union of all tables across specified keys.

        Args:
            dataset_fp (Path): Path to the HDF5 dataset.
            keys (List[str]): Keys that should be included in the result set.

        Returns:
            HDF5DataSource: DataSource
        """
        source = pd.HDFStore(dataset_fp, 'r')

        tables = []
        for key in keys:
            # FIXME: If _v_pathname is to be used on the outside, don't make it protected.
            tables += [node._v_pathname for node in source.get_node(str(key))]

        return HDF5DataSource(dataset_fp, tables=tables, source=source)

    def split(self, sizes: List[float], key: Optional[str]) -> List[HDF5DataSource]:
        """Partition the DataSource into N partitions. The size of each partition is defined by
        `sizes` parameter. Key defines how the DataSource is split.

        Args:
            sizes (List[float]): Defines size of new DataSource as a fraction of the old one.
            key (Optional[str]): When `None` the DataSource is split on the HDF5 key of each table.
                If specified, the value of metadata attribute key is used.

        Returns:
            List[HDF5DataSource]: List of DataSource partitions.
        """
        data_sources = []
        tables = self.tables
        n_tables = len(self.tables)
        for size in sizes[:-1]:
            if key is None:
                stratify = [table_key.rsplit('/', 1)[0] for table_key in tables]
            else:
                stratify = [
                    self.get_metadata(table_key)[key] for table_key in tables
                ]
            new_tables, tables = train_test_split(
                tables,
                train_size=int(n_tables*size),
                stratify=stratify
            )
            data_sources.append(
                HDF5DataSource(
                    self.dataset_fp,
                    new_tables,
                    self.source
                )
            )
        data_sources.append(
                HDF5DataSource(
                    self.dataset_fp,
                    tables,
                    self.source
                )
            )
        return data_sources
