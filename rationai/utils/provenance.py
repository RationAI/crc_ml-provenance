# Standard Imports
from pathlib import Path
from datetime import datetime
import json

from typing import Dict, List
import pandas as pd
import hashlib

# Third-party Imports
import xml.etree.ElementTree as ET

# Local Imports

class SummaryWriter:
    """SummaryWritter is a logger that uses dictionary as an internal storage
    data structure.

    Works on a singleton basis -- only one SummaryWriter per name.

    Raises:
        TypeError: Exception is raised if a named argument other than "value"
                   is passed to set() or add() methods.
        ValueError: Exception is raised if no keys are passed to set() or
                    add() methods.

    [Example Usage]

    sw = SummaryWriter.getLogger('provenance')
    print(sw.data)

    # { }

    sw.set('level1', 'level2', value='hello')
    print(sw.data)

    # {
    #     'level1': {
    #         'level2': 'hello'
    #     }
    # }

    sw.add('level1', 'level2-list', value=32)
    sw.add('level1', 'level2-list', value=15)
    print(sw.data)

    # {
    #     'level1': {
    #         'level2': 'hello',
    #         'level2-list': [32, 15]
    #     }
    # }

    """
    _instances = {}
    def __init__(self):
        self.clear()

    def set(self, *keys, **values):
        value = values.pop('value', None)
        if values:
            raise TypeError('Invalid parameters passed: {}'.format(str(values)))

        leaf = self.__get_leaf(*keys)
        leaf[keys[-1]] = value

    def get(self, *keys, **values):
        if values:
            raise TypeError('Invalid parameters passed: {}'.format(str(values)))
        leaf = self.__get_leaf(*keys)
        return leaf[keys[-1]]

    def add(self, *keys, **values):
        leaf = self.__get_leaf(*keys)
        value = values.pop('value', None)
        if values:
            raise TypeError('Invalid parameters passed: {}'.format(str(values)))
        if keys[-1] not in leaf:
            leaf[keys[-1]] = []
        leaf[keys[-1]].append(value)

    def __get_leaf(self, *keys):
        if len(keys) == 0:
            raise ValueError('At least one key must be provided.')
        node = self.data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        return node

    def now():
        return datetime.now().strftime('%d-%b-%Y %H:%M:%S')

    def hash_tables(hdfs_handler: Path, table_keys: List[str]) -> Dict:
        """Given a path to a pandas.HDFStore it calculates the sha256
        hash for each table as a combination of the table content and
        its associated metadata only.

        Args:
            filepath (Path): path to pandas.HDFStore
            groups (List[str]): list of group names

        Returns:
            Dict: dictionary of hashes grouped by groups
        """
        result = {}
        for idx, table_key in enumerate(table_keys):
            checksum = SummaryWriter.hash_table(hdfs_handler, table_key)
            result[f'table_{idx}_sha256'] = checksum
        return result

    def hash_table(hdfs_handler: pd.HDFStore, table_key: str) -> str:
        """Helper function for computing a hash for a single table.

        Args:
            hdfs_handler (pd.HDFStore): file handler to opened pd.HDFStore
            table_key (str): key for a table

        Returns:
            str: sha256 hash for a given table
        """
        sha256 = hashlib.sha256()

        # Table Checksum
        df = hdfs_handler.get(table_key)
        sha256.update(pd.util.hash_pandas_object(df).values)

        # Metadata Checksum
        d = hdfs_handler.get_storer(table_key).attrs.metadata
        [sha256.update(str.encode(repr(item), 'UTF-8')) for item in d.items()]

        checksum = sha256.hexdigest()
        return checksum

    def to_json(self, filepath: Path):
        with open(str(filepath), 'w') as json_log:
            json.dump(self.data, json_log, indent=True)

    def clear(self):
        self.data = {}
        self.vars = {
            'gen_counter': 0,
            'save_id': 0
        }

    @classmethod
    def getLogger(cls, name):
        if name not in cls._instances:
            new_sw = cls()
            cls._instances[name] = new_sw
        return cls._instances[name]
