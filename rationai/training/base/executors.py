from abc import ABC
from abc import abstractmethod

from rationai.utils.config import ConfigProto

class Executor(ABC):
    def __init__(self, config: ConfigProto):
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
