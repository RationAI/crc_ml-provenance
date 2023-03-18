# Standard Imports
import argparse
from pathlib import Path
import shutil

# Third-party Imports

# Local Imports
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class
from rationai.utils.provenance import SummaryWriter

from rationai.provenance import BUNDLE_TRAIN

sw_log = SummaryWriter.getLogger('provenance')
sw_log.clear()

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
        """
        self.__setup()
        self.__train()

    def __train(self):
        """Defines high-level training procedure.
        """
        train_gen = self.generators_dict[self.config.train_gen]

        valid_gen = None
        if self.config.valid_gen is not None:
            valid_gen = self.generators_dict[self.config.valid_gen]

        hist_log = self.executor.train(
            self.model,
            train_gen,
            valid_gen,
        )

        self.__process_results(hist_log)

    def __process_results(self, results):
        for metric, values in results.items():
            for idx, value in enumerate(values):
                if metric.startswith('val_'):
                    sw_log.set('iters', idx, 'metrics', 'valid', metric, value=value)
                else:
                    sw_log.set('iters', idx, 'metrics', 'train', metric, value=value)

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
        self.model.save_weights(Experiment.Config.experiment_dir / 'init.ckpt')
        sw_log.set('init_checkpoint_file', value=str(Experiment.Config.experiment_dir / 'init.ckpt'))

        # Build Executor
        executor_config = self.config.executor_class.Config(
            self.config.executor_config
        )
        executor_config.parse()
        self.executor = self.config.executor_class(executor_config)

    class Config(Experiment.Config):
        result_dir = None

        def __init__(self, json_dict: dict, eid: str):
            super().__init__(json_dict, eid)

            # Model Configuration
            self.model_class_name = None
            self.model_config = None

            # Executor Configuration
            self.executor_class_name = None
            self.executor_config = None

            # Generator Selection
            self.train_gen = None
            self.valid_gen = None

            # Datagen Configuration
            self.datagen_class = None
            self.datagen_config = None

        def parse(self):
            super().parse()

            definitions_config = self.config['definitions']
            configurations_config = self.config.get('configurations', dict())

            # Model Configuration
            self.model_class = get_class(definitions_config['model'])
            self.model_config = configurations_config.get('model', dict())

            # Executor Configuration
            self.executor_class = get_class(definitions_config['executor'])
            self.executor_config = configurations_config.get('executor', dict())

            # Generator Selection
            self.train_gen = self.config.get('train_generator')
            self.valid_gen = self.config.get('valid_generator', None)

            # Datagen Configuration
            self.datagen_class = get_class(definitions_config['datagen'])
            self.datagen_config = configurations_config.get('datagen', dict())


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    parser.add_argument('--eid', type=str, required=True, help='Experiment Identifier')
    args = parser.parse_args()

    json_filepath = args.config_fp
    sw_log.set('eid', value=args.eid)
    config = WSIBinaryClassifierTrain.Config.load_from_file(
        json_filepath=json_filepath,
        eid=args.eid
    )
    config.parse()
    WSIBinaryClassifierTrain(config).run()

    # Copy configuration file
    shutil.copy2(args.config_fp, Experiment.Config.experiment_dir / args.config_fp.name)
    sw_log.set('config_file', value=str(Path(Experiment.Config.experiment_dir / args.config_fp.name).resolve()))
    sw_log.set('script', value=__file__)
    sw_log.to_json((Experiment.Config.experiment_dir / BUNDLE_TRAIN).with_suffix('.log').resolve())

