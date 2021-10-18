# Standard Imports

# Third-party Imports

# Local Imports
from typing import Generator
from rationai.training.base.executors import Executor
from rationai.training.base.models import Model
from rationai.utils.config import ConfigProto
from rationai.utils.class_handler import get_class

class KerasExecutor(Executor):
    def __init__(self, config: ConfigProto):
        super().__init__(config)

    def train(self,
              model: Model,
              train_generator: Generator,
              valid_generator: Generator = None) -> dict:
        history = model.model.fit(x=train_generator,
            validation_data=valid_generator,
            epochs=self.config.epochs,
            max_queue_size=self.config.max_queue_size,
            workers=self.config.workers,
            use_multiprocessing=self.config.use_multiprocessing,
            callbacks=self.__get_callbacks(),
            verbose=1
        )
        return history.history

    def predict(self, model: Model, generator: Generator):
        return model.model.predict(x=generator,
            max_queue_size=self.config.max_queue_size,
            workers=self.config.workers,
            use_multiprocessing=self.config.use_multiprocessing,
            callbacks=self.__get_callbacks(),
            verbose=1
        )

    def __get_callbacks(self):
        return [callback_cls(**callback_cfg)
            for callback_cls, callback_cfg in zip(
                self.config.callback_classes,
                self.config.callback_configurations
            )
        ]

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.epochs = None

            # Keras Misc Settings
            self.max_queue_size  = None
            self.workers = None
            self.use_multiprocessing = None
            self.callback_classes = None
            self.callback_configurations = None

        def parse(self):
            # Training Params
            self.epochs = self.config.get('epochs', None)

            # Keras Misc Settings
            self.max_queue_size = self.config.get('max_queue_size', 1)
            self.workers = self.config.get('workers', 1)
            self.use_multiprocessing = self.config.get('use_multiprocessing', False)

            # Callback Parsing
            callback_config = self.config.get(
                'callbacks',
                {'definitions': {}, 'configurations': {}}
            )
            self.callback_classes = [
                get_class(callback) for callback in callback_config.get(
                    'definitions',
                    list()
                )
            ]
            self.callback_configurations = callback_config.get(
                'configurations',
                [dict()] * len(self.callback_classes)
            )