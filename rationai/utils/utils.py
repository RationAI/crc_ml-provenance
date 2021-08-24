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
from typing import Callable, List, NoReturn, Tuple, Any
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


def callable_has_signature(func: Callable, param_names: List[str]) -> bool:
    """
    Check any callable for the list of its parameter names.

    Parameters
    ----------
    func : Callable
        Any function or method to check.
    param_names : List[str]
        The list of parameter names checked against the signature of
        `func`.

    Return
    ------
    bool
        True if `param_names` contains exactly the parameter names of
        `func` (ignoring order), False otherwise.

    Raise
    -----
    TypeError
        When `func` is not Callable.
    """
    callable_params = list(inspect.signature(func).parameters)

    return (
            len(callable_params) == len(param_names)
            and all((name in callable_params) for name in param_names)
    )


def class_has_method(cls_object: type, method_name: str) -> bool:
    """
    Check whether a class object declares a method of given name.

    Illustrative example:

        class Dog:
            def __init__(self, age, color):
                self.age = age
                self.color = color

            def bark(self):
                return "Woof! " * self.age

        class_has_method(Dog, 'bark') => True
        class_has_method(Dog, 'meow') => False

    Parameters
    ----------
    cls_object : type
        The class object to check.
    method_name : str
        The name of the method to search `cls_object` for.

    Return
    ------
    bool
        True if `cls_object` has a method named `method_name`, False
        otherwise.
    """
    return (
        hasattr(cls_object, method_name)
        and callable(getattr(cls_object, method_name))
    )


def class_has_classmethod(cls_object: type, method_name: str) -> bool:
    """
    Check whether a class object declares a classmethod of given name.

    The same as `class_has_method`, with the additional constraint that the
    method is a class method.

    Parameters
    ----------
    cls_object : type
        The class object to check.
    method_name : str
        The name of the method to search `cls_object` for.

    Return
    ------
    bool
        True if `cls_object` has a classmethod named `method_name`, False
        otherwise.
    """
    return (
        class_has_method(cls_object, method_name)
        and type(cls_object.__dict__.get(method_name)) is classmethod
    )


def class_has_nonabstract_method(cls_object: type, method_name: str) -> bool:
    """
   Check whether a class object declares a non-abstract method of given name.

   The same as `class_has_method`, with the additional constraint that the
   method is not abstract method.

   Parameters
   ----------
   cls_object : type
       The class object to check.
   method_name : str
       The name of the method to search `cls_object` for.

   Return
   ------
   bool
       True if `cls_object` has a non-abstract method named `method_name`,
       False otherwise.
   """
    if not class_has_method(cls_object, method_name):
        return False

    method = getattr(cls_object, method_name)
    return getattr(method, '__isabstractmethod__', False) is False


def parse_module_and_class_string(descriptor: str) -> Tuple[str, str]:
    """
    Parse a class descriptor string into class and module descriptors.

    E.g.    'path.to.module.Class' -> ('path.to.module', 'Class')
            'Class' -> ('', 'Class')

    Ignores any dots surrounding the `descriptor`.

    Parameters
    ----------
    descriptor : str
        A class descriptor interpreted as the full path to a class.

    Return
    ------
    Tuple[str, str]
        A tuple containing the (module ID, class ID) strings.
    """
    dot_split = descriptor.strip('.').split('.')
    class_name = dot_split[-1]
    module_id = '.'.join(dot_split[:-1])
    return module_id, class_name


def load_class(class_descriptor: str) -> type:
    """
    Load a class by its class descriptor.

    Parameters
    ----------
    class_descriptor : str
        The full name of the class including its module namespace, e.g.
        'some.module.Class', where 'some.module' is the full module path and
        'Class' is the name of the class.

    Return
    ------
    Type
        The corresponding class.

    Raise
    -----
    AttributeError
        When the module on the corresponding module path exists, but does not
        define the expected class.
    ImportError
        When the module on the corresponding module path is nonexistent.
    """
    module_id, class_name = parse_module_and_class_string(class_descriptor)
    module = importlib.import_module(module_id)
    return getattr(module, class_name)


def run_method(obj: object, method_name: str, kwargs: dict) -> Any:
    """
    Runs a method of given object with passed parameters.

    Parameters
    ----------
    obj : object
        The object whose method should be run.
    method_name : str
        The name of the method to run.
    kwargs : dict
        The arguments to be passed to the method.

    Return
    ------
    Any
        The return value of the run method.

    Raise
    -----
    AttributeError
        When `obj` does not declare a (non-abstract) method `method_name`.
    """
    if not class_has_nonabstract_method(obj, method_name):
        raise AttributeError(
            f'object {obj} does not declare classmethod {method_name}'
        )
    return getattr(obj, method_name)(**kwargs)


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
