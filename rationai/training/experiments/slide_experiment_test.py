# Standard Imports
import argparse
import json

# Third-party Imports

# Local Imports
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class
from rationai.datagens.datagens import Datagen
from rationai.utils.config import ConfigProto


class WSIBinaryClassifierTest(Experiment):
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
        test_gen = self.generators_dict[self.config.test_gen]
        test_gen.set_batch_size(self.config.batch_size)

        while test_gen.sampler.active_node is not None:
            net_predicts = self.executor.predict(
                self.model,
                test_gen
            )
            self.__save_predictions(net_predicts)
            test_gen.sampler.next()
            test_gen.on_epoch_end()

    def __setup(self):
        """Builds components necesary for experiment.
            1. Datagen
            2. Model
            3. Executor
        """
        # Build Datagen
        datagen_config = self.config.datagen_class.Config(
            self.config.datagen_config
        )
        datagen_config.parse()
        self.generators_dict = self.config.datagen_class(datagen_config) \
            .build_from_template()

        # Build Model
        model_config = self.config.model_class.Config(
            self.config.model_config
        )
        model_config.parse()
        self.model = self.config.model_class(model_config)
        self.model.compile_model()

        # Build Executor
        executor_config = self.config.executor_class.Config(
            self.config.executor_config
        )
        executor_config.parse()
        self.executor = self.config.executor_class(executor_config)

    def __save_predictions(self, predictions):
        """Saves predictions in a file.

        Args:
            predictions (pandas.DataFrame): Network predictions for a slide
        """
        pass

    class Config(ConfigProto):
        def __init__(self, json_dict):
            super().__init__(json_dict)

            self.batch_size = None

            # Model Configuration
            self.model_class_name = None
            self.model_config = None

            # Executor Configuration
            self.executor_class_name = None
            self.executor_config = None

            # Generator Selection
            self.test_gen = None

            # Datagen Configuration
            self.datagen_class = None
            self.datagen_config = None

        def parse(self):
            self.batch_size = self.config.get('batch_size')

            definitions_config = self.config['definitions']
            configurations_config = self.config.get('configurations', dict())

            # Model Configuration
            self.model_class = get_class(definitions_config['model'])
            self.model_config = configurations_config.get('model', dict())

            # Executor Configuration
            self.executor_class = get_class(definitions_config['executor'])
            self.executor_config = configurations_config.get('executor', dict())

            # Generator Selection
            self.test_gen = self.config.get('test_generator')

            # Datagen Configuration
            self.datagen_class = get_class(definitions_config['datagen'])
            self.datagen_config = configurations_config.get('datagen', dict())


if __name__=='__main__':
    json_filepath = ...
    config = WSIBinaryClassifierTest.Config(
        json_dict=None,
        json_filepath=json_filepath
    )
    WSIBinaryClassifierTest(config).run()

