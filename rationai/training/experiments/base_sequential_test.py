# Standard Imports
import argparse
import json

# Third-party Imports

# Local Imports
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class
from rationai.datagens.datagens import Datagen
from rationai.utils.config import ConfigProto


class BaseSequentialTest(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.generators_dict = None
        self.executor = None
        self.model = None

    def run(self):
        """Base Sequential Test Experiment
                1. Loads data
                2. Processes every leaf node of a sampling tree.
        """
        self.__setup()
        test_gen = self.generators_dict[self.config.test_gen]
        test_gen.set_batch_size(self.config.batch_size)

        while test_gen.sampler.active_node is not None:
            net_predicts = self.executor.predict(
                self.model,
                test_gen
            )
            self.save_predictions(net_predicts, test_gen)
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

    def save_predictions(self, predictions, test_gen):
        """Saves predictions in a file.

        Args:
            predictions (pandas.DataFrame): Network predictions for a slide
            test_gen (np.ndarray): Test generator used to generate this epoch
        """
        raise NotImplementedError


