# Standard Imports
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import secrets
import hashlib
import json

# Third-party Imports
import xml.etree.ElementTree as ET
import pandas as pd
import prov.model
import prov.dot
import pygit2


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

    def to_json(self, filepath: Path):
        with open(str(filepath), 'w') as json_log:
            json.dump(self.data, json_log, indent=True)

    def clear(self):
        self.data = {
            'git_commit_hash': pygit2.Repository('.').revparse_single('HEAD').hex
        }
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

def parse_log(filepath: Path) -> Dict:
    # Open Provenance Log
    with open(filepath, 'r') as log_in:
        log = json.load(log_in)

    # Insert config
    json_path = Path(log['config_file'])
    with open(json_path, 'r') as json_in:
        log['config'] = json.load(json_in)

    return log

def flatten_lists(cfg: Dict) -> Dict:
    for key, val in cfg.items():
        if isinstance(val, list):
            cfg[key] = flatten_list(val)
    return cfg

def flatten_list(l: List):
    if not isinstance(l, List):
        raise ValueError('Not a list')
    str_list = [str(x) for x in l]
    return ', '.join(str_list)

def flatten_dict(d: Dict, sep='_') -> Dict:
    """Flattens dictionary by concatenating the nested keys
       Source: https://stackoverflow.com/a/41801708/9734414    
    """
    return pd.json_normalize(d, sep=sep).to_dict(orient='records')[0]

def hash_tables_by_groups(filepath: Path, groups: List[str]) -> Dict:
    """Given a path to a pandas.HDFStore it calculates the sha256
    hash for each table as a combination of the table content and
    its associated metadata only.

    Args:
        filepath (Path): path to pandas.HDFStore
        groups (List[str]): list of group names

    Returns:
        Dict: dictionary of hashes grouped by groups
    """
    hdfs = pd.HDFStore(path=filepath, mode='r')

    G_tables = {}
    for g_name in groups:
        G_tables[g_name] = {}
        for idx, node in enumerate(hdfs.get_node(f'/{g_name}')):
            checksum = hash_table(hdfs, node._v_pathname)
            G_tables[g_name][f'table_{idx}_sha256'] = checksum

    hdfs.close()
    return G_tables

def hash_tables_by_keys(hdfs_handler: pd.HDFStore, table_keys: List[str]) -> Dict:
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
        checksum = hash_table(hdfs_handler, table_key)
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

def export_to_image(bundle: prov.model.ProvBundle, name: str) -> None:
    """Create a png image from a bundle and suffix it with given name.
    
    Args: 
        bundle: The bundle of which the provenance is to be exported.
        name: The name of the bundle as a string, given a suffix to the filename
              of the image.
    """
    dot = prov.dot.prov_to_dot(bundle)
    dot.write_png(f"prov-{name}.png")

def get_sha256(filepath: str, mock_env: bool = False) -> str:
    """Generate a SHA256 hash of a file with a given filename.

    For the purpose of this thesis it can generate a random string when the given path is
    is not valid. If the script would be really deployed on server it would not be necessary.
    Adjusted example from:
    https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html

    Args:
        filename: The path to the file whose hash is to be calculated.
        mock_env: Whether the provenance is to be generated in a simulated mocked environment.

    Returns:
        Calculated hash value of the file or a random hash if the file does not exist and the function
        was called in a simulated mocked environment.

    Raises:
        FileNotFoundError: If the file does not exist and the function was not called in a simulated
        mocked enviroment.
    """
    if filepath is None:
        return None

    filepath = Path(filepath)
    sha256_hash = hashlib.sha256()

    if not filepath.exists():
        if mock_env:
            # For Keras checkpoint files
            return secrets.token_urlsafe(32)
        else:
            raise ValueError('File does not exist')
    elif filepath.is_dir():
        for filepath_i in filepath.glob('*'):
            sha256_hash.update(bytes.fromhex(get_sha256(filepath_i.resolve())))
        return sha256_hash.hexdigest()
    elif filepath.is_file():
        with open(filepath, "rb") as f:
            # read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
    else:
        raise ValueError('Unknown file type.')