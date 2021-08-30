# TODO: Describe all keys used by the pipeline in one place.
"""
An overview of keywords that are used to identify various types of paths.

Keys
----
results
    A folder where a run creates a foldler for all its results.

data_root
    A folder contaning a data set.
    I.e., a directory containing subfolders with raw files,
    processed dataframes, datasets, etc. for a collection of data.
"""

import logging
from pathlib import Path

from typing import Any
from typing import Dict
from typing import NoReturn
from typing import Optional

log = logging.getLogger('dir-struct')


class DirStructure:
    """
    Pipeline's manager of path variables that is provided to each step.
    Paths can be read and appended using string keys that identify them
    inside the internal dictionary.

    Existing keys are read-only.
    Trying to change such key results in an exception being raised.
    """
    paths: Dict[str, Path]

    def __init__(self):
        self.paths = dict()

    def add(self, key: str, path: Path, create=False) -> Path:
        """
        Adds a new key-path pair to the internal data structure.

        Parameters
        ----------
        key : str
            A string key that identifies its respective path.

        path : Path
            A key's value that identifies a disk location or a resource.

        create : bool
            If set to true, an empty directory is created from the path
            (including parent directories).

        Return
        ------
        Path
            Path added to the structure.

        Raise
        -----
        ValueError
            When `key` already exists inside `self.paths`.

        RAI_UNTESTABLE - tests do not cover whether a new folder is correctly
                         created when parameter `create` is True

        """
        if key in self.paths:
            raise ValueError(f'Cannot add key "{key}" to DirStructure: '
                              'the key already exists.')

        self.paths[key] = path

        if create and not path.exists():
            path.mkdir(parents=True)

        return Path(path)

    def get(self, key: str) -> Optional[Path]:
        """
        Returns a path identified by 'key'.

        Parameters
        ----------
        key : str
            A string key that identifies its respective path.

        Returns
        -------
        Path
            A path identified by the parameter key.
        """
        if key not in self.paths:
            log.info(f'Key "{key}" not found in DirStructure')
            return None
        return self.paths[key]

    def log_paths(self) -> NoReturn:
        # TODO: implement using Loggable decorator
        # for key, path in self.paths.items():
        #     summary_writer.set_value('paths', key, value=str(path))
        pass

