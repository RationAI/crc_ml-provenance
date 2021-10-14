# Standard Imports
import argparse
import json

# Third-party Imports

# Local Imports
from histopat.rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class
from rationai.datagens.datagens import Datagen

class WSIBinaryClassifierTrain(Experiment):
    def __init__(self, config):
        self.config = config
        self.generators_dict = None
        self.executor = None
        self.model = None

    def run(self):
        """WSI Binary Classifer
                1. Loads data
                2. Trains on entire training dataset
                3. Slide-wise testing - every slide is used independently
                    for testing - results in N output files for N test slides.
                    It is assumed that the leaves of the sampling tree are
                    entire whole slides.
        """
        self.__setup()
        self.__train()

    def __train(self):
        """Defines high-level training procedure.
        """
        train_gen = self.generators_dict[self.config.train_gen]
        valid_gen = self.generators_dict[self.config.valid_gen]
        _ = self.executor.fit(
            self.model,
            train_gen,
            valid_gen,
            **self.config.train_config
        )

    def __setup(self):
        """Builds components necesary for experiment.
            1. Datagen
            2. Model
            3. Executor
        """
        datagen_config = Datagen.Config(self.config.config_filepath)
        self.generators_dict = Datagen(datagen_config).build_from_template()

        model_class = get_class(self.config.model_class_name)
        self.model = model_class.build(**self.config.model_config)

        executor_class = get_class(self.config.executor_class_name)
        self.executor = executor_class(**self.config.executor_config)

    def __save_predictions(self, predictions):
        """Saves predictions in a file.

        Args:
            predictions (pandas.DataFrame): Network predictions for a slide
        """
        raise NotImplementedError

    class Config:
        def __init__(self, json_filepath, json_dict=None):
            super().__init__(json_filepath, json_dict)
            self.experiment_class = None

            # Model Configuration
            self.model_class_name = None
            self.model_config = None

            # Executor Configuration
            self.executor_class_name = None
            self.executor_config = None

            # Generator Selection
            self.train_gen = None
            self.valid_gen = None

            # ML Configuration
            self.train_config = None

        def __parse(self):
            # TODO: Set json-paths for values
            self.experiment_class = None

            # Model Configuration
            self.model_class_name = None
            self.model_config = None

            # Executor Configuration
            self.executor_class_name = None
            self.executor_config = None

            # Generator Selection
            self.train_gen = None
            self.valid_gen = None

            # ML Configuration
            self.train_config = None


if __name__=='__main__':
    json_filepath = ...
    config = WSIBinaryClassifierTrain.Config(
        json_dict=None,
        json_filepath=json_filepath
    )
    WSIBinaryClassifierTrain(config).run()

