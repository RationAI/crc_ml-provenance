import sys
import json
import copy
import threading
import importlib
import inspect
import numpy as np
import shutil

from enum import Enum
from pathlib import Path
from typing import NoReturn
from typing import Optional


class Mode(Enum):
    Train = 'train'
    Validate = 'validate'
    Test = 'test'
    Eval = 'evaluate'


class ExperimentLevel(Enum):
    DEBUG = 'debug'
    SEARCH = 'search'
    TEST = 'test'
    FINAL = 'final'


def divide_round_up(n, d):
    return (n + (d - 1)) // d


def mkdir(path: Path, parents=True) -> Path:
    if not path.exists():
        path.mkdir(parents=parents, exist_ok=True)
    return path


def load_from_module(identifier: str, *args, **kwargs):
    """Loads an object from a module given by identifier.

    Uses kwargs for object initializaiton.

    Possible abbreviations in identifier:
        tensorflow  ->  tf
        keras       ->  k, K

    Example:
        load_from_module('tf.K.optimizers.Adam', lr=0.0001)
    """

    # handle possible abbreviations
    if identifier.startswith(('k.', 'K.')):
        identifier = 'tensorflow.keras.' + identifier[2:]
    elif identifier.startswith('tf.'):
        identifier = identifier.replace('tf.', 'tensorflow.')
    identifier = identifier.replace('.k.', '.keras.')
    identifier = identifier.replace('.K.', '.keras.')

    class_name = identifier.split('.')[-1]
    module_name = identifier.replace(f'.{class_name}', '')

    # _class = eval(f'importlib.import_module(module).{class_name}')
    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c(*args, **kwargs)


def get_module_class_names(module: str):
    """Returns a list of all classes available in the given module"""
    if not module or module not in sys.modules:
        print(f'{module} not found in sys.modules')
        return None
    return [class_name for class_name, _ in
            inspect.getmembers(sys.modules[module], inspect.isclass)]


def join_module_path(module: str, class_name: str):
    """Returns modular path to a class.
    Returns None if there is no such class inside the module.

        string: module          e.g. tensorflow.keras.optimizers
        string: class_name      e.g. Adam
    """
    if class_name in get_module_class_names(module):
        return f'{module}.{class_name}'
    return None


def json_to_dict(filepath: Path) -> Optional[dict]:
    """Reads JSON and returns a dictionary or None upon failure."""
    try:
        with filepath.open('r') as f_summary:
            return json.load(f_summary)
    except Exception:
        return None


def merge_dicts(a: dict, b: dict) -> dict:
    """Returns a deepcopy of two merged dictionaries."""
    a_copy = copy.deepcopy(a)
    b_copy = copy.deepcopy(b)
    return _merge_dicts_recursive(a_copy, b_copy)


def _merge_dicts_recursive(a: dict, b: dict, path: list = []) -> dict:
    """Returns a merged dictionary
    that contains all the nested values from both input dictionaries.

    Raises an error when a conflict is encountered.
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dicts_recursive(a[key], b[key], path + [str(key)])
            elif a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key])):
                pass  # same leaf value
            else:
                raise ValueError('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def make_archive(source: Path, target: Path, format: str = 'zip') -> NoReturn:
    """Compresses 'source' and stores it as 'target'

    Parameters:
        format: one of "zip", "tar", "gztar", "bztar", or "xztar"
    """
    if isinstance(source, str):
        source = Path(source)

    archive_path = shutil.make_archive(
        source, format=format, root_dir=source.parent, base_dir=source.name)

    shutil.move(archive_path, target)


def detect_file_format(folder: Path,
                       pattern: str,
                       extensions: list,
                       verbose: bool = False) -> Path:
    """Auto detects file format for "folder/pattern".

    Used if a file stem or prefix is known, but the extension is not.

    If multiple files with the same stem are found,
    then extensions defines search priority for best result.

    Exception is raised if no suitable file was found.
    """
    detected_files = [p for p in folder.glob(pattern)]

    if len(detected_files) == 1:
        return detected_files[0]

    if len(detected_files) == 0:
        raise ValueError(f'No files found in: {folder}/{pattern}')

    if len(detected_files) > 1:
        suffixes = list(map(lambda p: str(p).split('.')[-1], detected_files))
        if verbose:
            print(f'Multiple extensions ({suffixes}) found for {folder}/{pattern}')

        for ext in extensions:
            if ext in suffixes:
                return [p for p in detected_files if str(p).endswith(ext)][0]

    raise ValueError(f'Unexpected file formats encountered: {detected_files}')


class ThreadSafeIterator:
    """Iterator with a lock for multiprocessing.
    Allows usage of generator with pool of workers"""

    def __init__(self, alist):
        self.data = alist
        self.idx = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        with self.lock:
            try:
                res = self.data[self.idx]
                self.idx += 1
                return res
            except Exception:
                raise StopIteration

    def len(self):
        return self.__len__()

    def next(self):
        return self.__next__()
