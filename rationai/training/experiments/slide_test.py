# Standard Imports
from pathlib import Path
import argparse
import shutil

# Third-party Imports
import pandas as pd
import numpy as np

# Local Imports
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class
from rationai.utils.provenance import SummaryWriter
from rationai.utils.provenance import hash_tables_by_keys
from rationai.utils.provenance import hash_table

from rationai.provenance import BUNDLE_EVAL

sw_log = SummaryWriter.getLogger('provenance')

class WSIBinaryClassifierTest(Experiment):
    """Provides predictions for each slide.

    Important:
        - SamplerTree data structure is used
        - Each leaf node of a sampler tree consists of a single slide
    """
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.__setup()
        
        # Predictions
        self.hdfstore_output = pd.HDFStore(
            Experiment.Config.experiment_dir / 'predictions.h5',
            'a'
        )
        
        test_gen = self.generators_dict[self.config.test_gen]
        
        hashes = hash_tables_by_keys(
            hdfs_handler=test_gen.sampler.data_source.source,
            table_keys=test_gen.sampler.data_source.tables
        )
        sw_log.set('DEBUG', 'before', value=hashes)
        
        while test_gen.sampler.active_node is not None:
            net_predicts = self.executor.predict(
                self.model,
                test_gen
            )
            metrics = self.evaluate(net_predicts, test_gen)
            self.save_predictions(net_predicts, metrics, test_gen)
            test_gen.sampler.next()
            test_gen.on_epoch_end()
            sw_log.vars['gen_counter'] += 1
        
        hashes = hash_tables_by_keys(
            hdfs_handler=test_gen.sampler.data_source.source,
            table_keys=test_gen.sampler.data_source.tables
        )
        sw_log.set('splits', test_gen.name, value=hashes)
        sw_log.set('DEBUG', 'after', value=hashes)
        self.hdfstore_output.close()
        sw_log.set('predictions', 'prediction_file', value=str(Experiment.Config.experiment_dir / "predictions.h5"))

    def evaluate(self, net_predicts, gen):
        metrics = {}
        y_true = [s.entry['is_cancer'] for s in gen.epoch_samples]
        
        for evaluator in self.evaluators:
            evaluator.update_state_fn(y_pred=np.ravel(net_predicts), y_true=np.ravel(y_true))
            metrics[evaluator.name] = evaluator.result()
            evaluator.reset_state()
        return metrics
        
    def save_predictions(self, predictions, metrics, test_gen):
        """Saves predictions in a file.

        Args:
            predictions (pandas.DataFrame): Network predictions for a slide
        """
        # Get Output Data
        output_data = test_gen.sampler.active_node.data.copy()
        output_data['pred'] = predictions

        output_metadata = test_gen.epoch_samples[0].metadata.copy()
        output_metadata['metrics'] = metrics
        output_table_key = test_gen.epoch_samples[0].entry['_table_key']

        # Save into HDFStore
        self.hdfstore_output.append(output_table_key, output_data)
        self.hdfstore_output \
            .get_storer(output_table_key) \
            .attrs \
            .metadata = output_metadata

        hash_val = hash_table(
            hdfs_handler=self.hdfstore_output,
            table_key=output_table_key
        )
        for metric_name, metric_value in metrics.items():
            sw_log.set('evaluations', hash_val, metric_name, value=float(metric_value))
        sw_log.set('predictions', 'sha256', f'table_{sw_log.vars["gen_counter"]}_sha256', value=hash_val)
        
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
        
        # Build Evaluators
        self.evaluators = []
        for evaluator_class_name, evaluator_config_dict in zip(
            self.config.evaluator_classes, self.config.evaluator_configs
        ):
            evaluator_class = get_class(evaluator_class_name)
            evaluator_config = evaluator_class.Config(evaluator_config_dict)
            evaluator_config.parse()
            self.evaluators.append(evaluator_class(evaluator_config))
            

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
            self.test_gen = None

            # Datagen Configuration
            self.datagen_class = None
            self.datagen_config = None
            
            # Evaluators Configuration
            self.evaluator_classes = None
            self.evaluator_configs = None

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
            self.test_gen = self.config.get('test_generator')

            # Datagen Configuration
            self.datagen_class = get_class(definitions_config['datagen'])
            self.datagen_config = configurations_config.get('datagen', dict())
            
            # Evaluators Configuration
            self.evaluator_classes = definitions_config.get('evaluators', list())
            self.evaluator_configs = configurations_config.get(
                'evaluators',
                [dict()] * len(self.evaluator_classes)
            )

if __name__=='__main__':
    sw_log.clear()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    parser.add_argument('--eid', type=str, required=True, help='Experiment Identifier')
    args = parser.parse_args()

    json_filepath = args.config_fp
    sw_log.set('eid', value=args.eid)
    config = WSIBinaryClassifierTest.Config.load_from_file(
        json_filepath=json_filepath,
        eid=args.eid
    )
    config.parse()
    WSIBinaryClassifierTest(config).run()
    sw_log.set('script', value=__file__)

    # Copy configuration file
    shutil.copy2(args.config_fp, Experiment.Config.experiment_dir / args.config_fp.name)
    sw_log.set('config_file', value=str(Path(Experiment.Config.experiment_dir / args.config_fp.name).resolve()))
    sw_log.set('script', value=__file__)
    sw_log.to_json((Experiment.Config.experiment_dir / BUNDLE_EVAL).with_suffix('.log').resolve())
