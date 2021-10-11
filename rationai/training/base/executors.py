from abc import ABC
from abc import abstractmethod

class Executor(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
