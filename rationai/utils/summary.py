import git
import json
import logging
import copy
from datetime import datetime
from typing import Optional
from typing import Type

from .utils import json_to_dict
from .utils import merge_dicts


class SummaryWriter:
    """Class responsible for logging to a JSON log file"""
    def __init__(self, params: dict, experiment_goal: Optional[str]):
        self.experiment_log = dict()
        self.summary_path = None
        self.log = logging.getLogger('summary')

        repo = git.Repo(search_parent_directories=True)
        self.set_value('git_branch', value=str(repo.active_branch))
        self.set_value('git_commit_sha', value=repo.head.object.hexsha)

        self.set_value('params', value=copy.deepcopy(params))
        self.set_value('description', value=experiment_goal)

    def set_path(self, summary_path):
        """Sets log file path"""
        self.summary_path = summary_path

    def update_log(self):
        """Writes in-memory log to disk."""
        try:
            self.log.debug('Writing log to disk.')
            # with open(str(self.summary_path), 'w') as json_log:
            #     json.dump(self.experiment_log, json_log, indent=True)

            # Merge added because continue experiment causes troubles
            # when different components set values and update SW.
            # TODO: lock exclusive access for this IO? (portalocker)
            summary = json_to_dict(self.summary_path)
            if summary is None:
                summary = self.experiment_log
            else:
                summary = merge_dicts(summary, self.experiment_log)
            with open(str(self.summary_path), 'w') as json_log:
                json.dump(summary, json_log, indent=True)

        except AttributeError:
            self.log.warn('SummaryWriter has no output filepath specified.')
            raise ValueError('Call SummaryWriter.set_path(path) '
                             'before saving the summary.')

    def set_value(self, *keys, **values):
        """Sets a key value pair to a log

        Args:
            keys : JSON serializble type
                Variadic arguments that form a nested key structure

            value : JSON serializable type
                A value stored in the nested key structure.
        """
        value = values.pop('value', None)
        if values:
            raise TypeError(f'Invalid parameters passed: {str(values)}')

        leaf = self._get_leaf(*keys)
        leaf[keys[-1]] = value

    def add_value(self, *keys, **values):
        leaf = self._get_leaf(*keys)
        value = values.pop('value', None)
        if values:
            raise TypeError(f'Invalid parameters passed: {str(values)}')

        if keys[-1] not in leaf:
            leaf[keys[-1]] = []
        leaf[keys[-1]].append(value)

    def _get_leaf(self, *keys):
        if len(keys) == 0:
            raise ValueError('At least one key must be provided.')

        node = self.experiment_log
        for key in keys[:-1]:
            node = node.setdefault(key, {})

        return node

    @staticmethod
    def now(strftime: str = '%d-%b-%Y %H:%M:%S') -> Type[datetime]:
        """Returns now() in datetime.

        Args:
            strftime : str
                Specifies datetime format.
        """
        return datetime.now().strftime(strftime)
