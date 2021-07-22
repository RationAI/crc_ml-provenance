import logging
from copy import deepcopy
from pathlib import Path

from .summary import SummaryWriter

log = logging.getLogger('paths')

# TODO: maybe differentiate between files and dirs -> 2 pairs of dicts and methods


class DirStructure:
    """Holds directory structure as a dictionary of Paths.
    Its instance is passed through experiment steps.

    'results'   - experiment results are written to this folder
                - usually <project>/models

    'data_root' - experiment data are read from this folder
                - should contain input(rgb), label, bg, coords_maps
                  and datasets subfolders

    Keys 'input', 'label', 'coord_maps', 'dataset' specify respective locations
    inside 'data_root' folder.

    Modules can manipulate and add new items to
    """
    def __init__(self, data_dirs: dict):
        self.data_dirs_params = data_dirs
        self.directories = dict()

        # Set by ExperimentInitializer
        self.eid = None

        # Experiment writes results to:
        self.add('results', Path(data_dirs.get('results', 'models')))

        # Experiment takes data from:
        data_root = Path(data_dirs['root'])
        self.add('data_root', data_root)

        # Repetition in select looks weird. Fix later
        self.add('input', data_root / self._select('rgb', default_dir='rgb'))
        self.add('label', data_root / self._select('label', default_dir='label'))

        # expects the new dir strcuture where
        # dataset has the same name stem as coord_maps' dir
        # (old structure has to supply full paths)
        dataset_name = data_dirs['dataset']['name']
        self.add('dataset', data_root / 'datasets' / dataset_name)  # is a file

        if 'coord_maps' in data_dirs:
            self.add('coord_maps', data_root / 'coord_maps' / data_dirs['coord_maps'])
        else:
            cm_path = data_root / 'coord_maps' / dataset_name.split('.')[0]
            if not cm_path.exists():
                raise ValueError("Please specify 'coord_maps' in the configuration file. "
                                 f"{cm_path} does not exist.")
            self.add('coord_maps', cm_path)

    def add(self, key: str, path: Path, create=False) -> Path:
        """Adds new path to data structure under key 'key'
        If key exists, new path overrides the previous path"""

        if key in self.directories:
            if self.directories[key] == path:
                return Path(path)
            log.debug(f"Existing path in DirStructure changed. \
                     key='{key}' path: {self.get(key)} -> {path}")

        self.directories[key] = path

        # last guard allows mkdir only inside current experiment folder
        if create and not path.exists() and '.' not in str(path.name) and \
                self.eid in path.parts:
            path.mkdir(parents=True)

        return Path(path)

    def get(self, key: str) -> Path:
        if key not in self.directories:
            log.debug(f"No key '{key}' found in DirStructure")
            return None
        return deepcopy(self.directories[key])

    def create_dirs(self):
        """Creates the directory structure but ignores specific keys"""
        # Ignores following entries in the structure
        IGNORE_KEYS = ['data_root', 'input', 'label', 'dataset', 'coord_maps']

        for path_key in self.directories:
            if path_key in IGNORE_KEYS:
                continue
            path = self.directories[path_key]
            if not path.is_file() and not path.exists():
                path.mkdir(parents=True)

    def log_paths_to_summary(self,
                             summary_writer: SummaryWriter,
                             update_log=True) -> SummaryWriter:
        """Logs all paths found in self.directories to SummaryWriter.
        Also calls sw.update_log() if update_log is True."""
        for key, path in self.directories.items():
            summary_writer.set_value('paths', key, value=str(path))

        if update_log:
            summary_writer.update_log()
        return summary_writer

    def _select(self, config_key: str, default_dir: str):
        """If config_key exists in config file, its value is returned.
        Otherwise default_dir is used."""
        if default_dir not in self.data_dirs_params:
            return default_dir

        return self.data_dirs_params[config_key]
