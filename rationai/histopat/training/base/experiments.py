# Standard Imports
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

# Local Imports
from rationai.histopat.utils.config import ConfigProto

# Third-party Imports


class Experiment(ABC):
    def __init__(self, config: ConfigProto):
        self.config = config

    @abstractmethod
    def run(self):
        pass

    class Config(ConfigProto):
        """Experiment.Config is a special type of config that maintains
        static variable "experiment_dir". This variable sets up a context for
        every submodule of an experiment. That way a dynamically created
        output directory can be propagated and referenced from any module.

        Anytime an input file is expected and NoneType is obtained, the path
        should be substituted with Experiment.Config directory.
        """
        experiment_dir = None

        def __init__(self, json_dict: dict, eid: str):
            super().__init__(json_dict)
            self.eid = eid
            self.output_dir = None

        @classmethod
        def load_from_file(cls, json_filepath: Path, eid: str) -> Experiment.Config:
            with open(json_filepath, 'r') as json_finput:
                json_config = json.load(json_finput)
            return cls(json_config, eid)

        def __set_experiment_dir(self):
            Experiment.Config.experiment_dir = self.output_dir / self.eid
            Experiment.Config.experiment_dir.mkdir(parents=True, exist_ok=True)

        def parse(self):
            assert self.eid is not None, "Set EID first."
            self.output_dir = Path(self.config['output_dir']).resolve()
            self.__set_experiment_dir()
