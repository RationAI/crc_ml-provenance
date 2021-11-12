from abc import ABC, abstractmethod

from rationai.histopat.utils.config import ConfigProto


class Executor(ABC):
    def __init__(self, config: ConfigProto):
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
