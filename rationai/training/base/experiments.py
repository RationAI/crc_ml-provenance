# Standard Imports
from abc import ABC
from abc import abstractmethod

# Third-party Imports

# Local Imports
from rationai.utils.config import ConfigProto

class Experiment(ABC):
    def __init__(self, config: ConfigProto, eid: str):
        self.config = config
        self.eid = eid

    @abstractmethod
    def run(self):
        pass
