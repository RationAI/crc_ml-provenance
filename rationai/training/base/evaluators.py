from abc import ABC, abstractmethod

from rationai.utils.config import ConfigProto
from rationai.utils.class_handler import get_class

class Evaluator(ABC):
    def __init__(self, config: ConfigProto):
        self.config = config
        self.name = self.config.name
        self.reset_state()

    @abstractmethod
    def reset_state(self):
        raise NotImplementedError

    def update_state(self, input_dict):
        self.update_state_fn(
            **{
                expected_name: input_dict[actual_name]
                for expected_name, actual_name in self.config.mapping.items()
            }
        )

    @abstractmethod
    def update_state_fn(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def result(self):
        raise NotImplementedError

    class Config(ConfigProto):
        def __init__(self, json_dict):
            super().__init__(json_dict)
            self.name = None
            self.mapping = None
            self.threshold = None

        def parse(self):
            self.name = self.config.get('name', __name__)
            self.mapping = self.config.get('mapping', dict())
            self.threshold = self.config.get('threshold', 0.5)
