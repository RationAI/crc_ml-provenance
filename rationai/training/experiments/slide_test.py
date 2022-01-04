# Standard Imports
from pathlib import Path
import argparse

# Third-party Imports
import pandas as pd

# Local Imports
from rationai.training.experiments.base_sequential_test import BaseSequentialTest
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class

class WSIBinaryClassifierTest(BaseSequentialTest):
    """Provides predictions for each slide.

    Important:
        - SamplerTree data structure is used
        - Each leaf node of a sampler tree consists of a single slide
    """
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.hdfstore_output = pd.HDFStore(
            Experiment.Config.experiment_dir / 'predictions.h5',
            'a'
        )
        super().run()
        self.hdfstore_output.close()

    def save_predictions(self, predictions, test_gen):
        """Saves predictions in a file.

        Args:
            predictions (pandas.DataFrame): Network predictions for a slide
        """
        # Get Output Data
        output_data = test_gen.sampler.active_node.data
        output_data['pred'] = predictions

        output_metadata = test_gen.epoch_samples[0].metadata
        output_table_key = test_gen.epoch_samples[0].entry['_table_key']

        # Save into HDFStore
        self.hdfstore_output.append(output_table_key, output_data)
        self.hdfstore_output \
            .get_storer(output_table_key) \
            .attrs \
            .metadata = output_metadata

    class Config(Experiment.Config):
        result_dir = None

        def __init__(self, json_dict: dict, eid: str):
            super().__init__(json_dict, eid)
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
            super().parse()
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    parser.add_argument('--eid', type=str, required=True, help='Experiment Identifier')
    args = parser.parse_args()

    json_filepath = args.config_fp
    config = WSIBinaryClassifierTest.Config.load_from_file(
        json_filepath=json_filepath,
        eid=args.eid
    )
    config.parse()
    WSIBinaryClassifierTest(config).run()

