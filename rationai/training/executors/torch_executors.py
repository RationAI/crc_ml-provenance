# Standard Imports

# Third-party Imports

# Local Imports
from typing import Generator
from rationai.training.base.experiments import Experiment
from rationai.training.base.executors import Executor
from rationai.training.base.models import Model
from rationai.utils.config import ConfigProto, build_from_config
from rationai.utils.class_handler import get_class

class TorchExecutor(Executor):
    def __init__(self, config: ConfigProto):
        super().__init__(config)
        self.logger = build_from_config(self.config.logger)

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

    
    class Config(ConfigProto):
        
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.epochs: int
            

        def parse(self):
            # Training Params
            self.epochs = self.config.get('epochs', None)
           

        def __prepend_filepaths(self, config: dict):
            if 'filepath' in config:
                config['filepath'] = str(Experiment.Config.experiment_dir / config['filepath'])

    