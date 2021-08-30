import json
import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Type
from typing import Union

from rationai.utils import DirStructure
from rationai.utils import make_archive

log = logging.getLogger('exp-init')

# A relative path to the dir inside the project containing the source code
SOURCE_DIR = 'rationai'
# Following patterns will be ignored when logging source code
IGNORE_PATTERNS = [
    '.*',
    '*.pyc',
    '*.ipynb',
    '__pycache__'
]


class ExperimentInitializer:
    """The class initializes pipeline runs
    and archives the used code.

    If configured, it prepares an experiment follow-up.
    """
    def __init__(self,
                 exp_params: dict,
                 data_dirs_params: dict):

        self.exp_params = exp_params
        self.dir_struct = DirStructure()

        self.eid = self._generate_id()
        # Set in case we continue previous experiment
        self.continue_eid = None
        self.use_copy_mode = False

        self._init_dirs(data_dirs_params)
        if 'continue' in exp_params and exp_params['continue'].get('eid'):
            self._handle_continue_experiment()

        self._log_source_code()

        log.info(f'EXPERIMENT ID: {self.eid}')

        # Set final eid to DirStruct
        self.dir_struct.eid = self.eid

    def to_prev_path(self, path: Union[Path, str]) -> Path:
        """Replaces new eid in the path with previous experiment id."""
        if not self.continue_eid:
            return path
        return Path(str(path).replace(self.eid, self.continue_eid))

    def to_new_path(self, path: Union[Path, str]) -> Path:
        """Replaces eid in the previous path with the new one."""
        if not self.continue_eid:
            return Path(path)

        parts = list(Path(path).parts)
        if self.continue_eid not in parts:
            return Path(path)

        parts[parts.index(self.continue_eid)] = deepcopy(self.eid)
        parts[0] = '' if parts[0] == '/' else parts[0]
        return Path('/'.join(parts))

    def get_dir_struct(self) -> Type[DirStructure]:
        """Returns entire DirStructure dictionary containing pathlib Paths"""
        return self.dir_struct

    def get_path(self, key: str) -> Optional[Path]:
        """Returns a Path from DirStructure stored under 'key'"""
        return self.dir_struct.get(key)

    def add_path(self, key: str, path: Path, create=False) -> Path:
        return self.dir_struct.add(key, path, create)

    def _generate_id(self) -> str:
        """Generates id. Takes experiment continuation into account"""

        # If the experiment is run on metacentrum, let the experiment id
        # be the same as PBS Jobid to make pairing easier.
        if 'PBS_JOBID' in os.environ:
            return os.environ['PBS_JOBID']

        # Otherwise, generate random UUID for experiment with custom prefix
        eid_prefix = self.exp_params.get('experiment_prefix') or ''
        eid = str(uuid4())
        if eid_prefix:
            eid = eid_prefix + eid[8:]

        if 'continue' in self.exp_params and self.exp_params['continue'].get('eid'):
            eid = deepcopy(self.exp_params['continue']['eid'].strip('/')) \
                + f'--{eid_prefix}-{str(uuid4())[:5]}'

        return eid

    def _init_dirs(self, path_config: dict) -> NoReturn:
        """
        Adds basic generic paths to DirStructure instance.
        """
        # Input data root
        self.dir_struct.add('data_root', Path(path_config['root']))

        # Location where to create a folder for run results
        result_dir = self.dir_struct.add('results', Path(path_config.get('results', 'results')))

        # TODO: rename to `rundir` (globally in a separate commit)
        eval_dir = self.dir_struct.add('expdir', result_dir / self.eid)

        # TODO: move to a better place when refactoring this class (ckpts is not generic)
        self.dir_struct.add('checkpoints', eval_dir / 'callbacks' / 'ModelCheckpoint')

    def _log_source_code(self):
        """Saves the folder with source code to the experiment dir as a zip

        Many important details (e.g., model architecture) are missing
        in the summary file.
        """
        eval_dir = self.dir_struct.get('expdir')

        target_zip = eval_dir / f'{SOURCE_DIR}.zip'

        if target_zip.exists():
            log.debug(f'Removing old source code from: {target_zip}')
            os.unlink(target_zip)

        shutil.copytree(src=SOURCE_DIR,
                        dst=eval_dir / SOURCE_DIR,
                        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))

        make_archive(eval_dir / SOURCE_DIR, target_zip)
        shutil.rmtree(eval_dir / SOURCE_DIR)

        log.info(f'The current checkpoint of the source code has been logged: {target_zip}')

    def _handle_continue_experiment(self):
        self.continue_eid = self.exp_params['continue'].get('eid').strip('/')
        self.use_copy_mode = self.exp_params['continue'].get('copy_data', True)

        if not self.continue_eid:
            raise ValueError('To continue a previous experiment, specify its id in config')

        log.info(f'Continuing experiment {self.continue_eid} as {self.eid}')

        if self.use_copy_mode:
            log.info('Copy mode ON: Data will be copied to result dir of the current experiment')
            self._copy_experiment_results()
        else:
            log.info('Copy mode OFF: Results of the previous experiment '
                     'will be accessed without making copies')
            # TODO: implement
            raise Exception('Copy mode off is not implemented yet')

    def _copy_experiment_results(self):
        """Copies everything from the results dirs of the previous experiment.
        The previous summary.json file is renamed to summary-<prev_EID>.json"""
        if not self.to_prev_path(deepcopy(self.dir_struct.get('expdir'))).exists():
            raise ValueError(f'Cannot follow up on "{self.continue_eid}".'
                             'Experiment does not exist.')

        self._load_paths_from_summary()

        tar_eval = self.get_path('expdir')

        src_eval = self.to_prev_path(str(tar_eval))

        # NOTE: Python 3.8 allows dirs_exist_ok=True
        shutil.copytree(src_eval, tar_eval, ignore=shutil.ignore_patterns('summary*.json'))

        # Copy & rename summary.json
        shutil.copy2(src=src_eval / 'summary.json',
                     dst=tar_eval / f'summary-{self.continue_eid}.json', )

    def _load_paths_from_summary(self):
        path = self.to_prev_path(deepcopy(self.dir_struct.get('expdir'))) / 'summary.json'

        with path.open('r') as f_summary:
            prev_summary = json.load(f_summary)

        # NOTE: warn a user if no paths found?
        paths = prev_summary.get('paths', [])

        for path_key in paths:
            self.dir_struct.add(path_key, self.to_new_path(paths[path_key]), create=False)
