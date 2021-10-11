from abc import ABC
from abc import abstractmethod

class Experiment(ABC):

    @abstractmethod
    def run(self):
        pass
