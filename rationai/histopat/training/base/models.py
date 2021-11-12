from abc import ABC
from abc import abstractmethod

class Model:
    @abstractmethod
    def build_from_file(model_filepath):
        # Does saved model need to be compiled?
        pass

    @abstractmethod
    def load_weights(self):
        raise NotImplementedError

    @abstractmethod
    def compile(self):
        raise NotImplementedError