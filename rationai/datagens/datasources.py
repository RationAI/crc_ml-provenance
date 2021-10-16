"""
TODO: Missing docstring.
"""
# Standard Imports
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
from typing import Optional

# Third-party Imports
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.io.pytables import HDFStore

# Local Imports
from rationai.utils.config import ConfigProto


class DataSource(ABC):
    """Abstract class for DataSource. It defines required methods."""
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def get_table(self) -> pd.DataFrame:
        """Retrieves full dataset defined by this data source.

        Returns:
            pd.DataFrame: Full dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self, dataset_fp: Path, config: ConfigProto) -> Dict[str, DataSource]:
        """Instantiate datasources from source dataset file and supplied
        configuration. Returns dictionary of datasources.

        Args:
            dataset_fp (Path): Source dataset file.
            config (ConfigProto): Datasource configuration file.

        Returns:
            Dict[str, DataSource]: Dictionary of datasources.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, entry: dict) -> dict:
        """Retrieves metadata for a entry.

        Args:
            entry (dict): Entry as a dictionary.

        Returns:
            dict: Metadata
        """
        raise NotImplementedError

    @abstractmethod
    def split(self, sizes: List[float], key: Optional[str]) -> List[DataSource]:
        """Splits datasource into N parts, where N is len(sizes)==len(key).

        Args:
            sizes (List[float]): Defines size of the split dataset as a fraction
                of the original dataset. Must sum to one.
            key (Optional[str]): Key based on which to split the datasource.

        Returns:
            List[DataSource]: List of split datasources.
        """
        raise NotImplementedError


class HDF5DataSource(DataSource):
    """DataSource for loading HDF5 Storage Files"""
    def __init__(self):
        self.dataset_fp = None
        self.tables = None
        self.source = None

    def get_table(self) -> pd.DataFrame:
        """Retrieves table stored at a given table key path.

        Args:
            table_key (str): Path within a HDF5 file to a table.

        Returns:
            (pd.DataFrame): DataFrame stored at given path.
        """
        return pd.concat([
            self.source.select(table_key)
                .assign(_table_key=table_key)
            for table_key in self.source
        ])

    def get_metadata(self, entry: dict) -> dict:
        """Retrieves table metadata belonging to an entry from that table.
        The table key is stored automatically to a table on get_table() call.

        Args:
            entry (dict): Entry from a table.

        Returns:
            dict: Metadata from a table.
        """
        try:
            return self.source.get_storer(entry['table_key']).attrs.metadata
        except AttributeError:
            return {}

    @classmethod
    def load_dataset(self, dataset_fp: Path, config: ConfigProto) -> Dict[HDF5DataSource]:
        """Loads the dataset as a union of all tables across specified keys.

        Args:
            dataset_fp (Path): Path to dataset.
            config (ConfigProto): HDF5 DataSource ConfigProto

        Returns:
            Dict[HDF5DataSource]: Dictionary of datasets.
        """
        data_source = self()
        data_source.dataset_fp = dataset_fp

        source = pd.HDFStore(dataset_fp, 'r')
        data_source.source = source

        tables = []
        for key in config.keys:
            tables += [node._v_pathname for node in source.get_node(str(key))]
        data_source.tables = tables

        if len(config.names) == 1:
            return {config.names[0]: data_source}

        data_sources = data_source.split(
            sizes=config.split_probas,
            key=config.split_on
        )
        return dict(zip(config.names, data_sources))

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

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.dataset_fp = None
            self.keys = None
            self.names = None
            self.split_probas = None
            self.split_on = None

        def parse(self):
            self.dataset_fp = self.config['_data']
            self.keys = self.config['keys']
            self.names = self.config['names']
            self.split_probas = self.config['split_probas'] or [1.0]
            self.split_on = self.config['split_on']