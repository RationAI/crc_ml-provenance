# Standard Imports
from pathlib import Path
import argparse

# Third-party Imports
import pandas as pd
import numpy as np

# Local Imports
from rationai.training.experiments.base_sequential_test import BaseSequentialTest
from rationai.training.experiments.slide_test import WSIBinaryClassifierTest
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class

class SegmentationTest(BaseSequentialTest):
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
        output_data = test_gen.sampler.active_node.data

        output_metadata = test_gen.epoch_samples[0].metadata
        output_table_key = test_gen.epoch_samples[0].entry['_table_key']

        seg_result_dir = Experiment.Config.experiment_dir / 'segmentations'
        if not seg_result_dir.exists():
            seg_result_dir.mkdir(parents=True)
        seg_result_fp = seg_result_dir / Path(output_metadata['slide_fp']).stem
        np.save(seg_result_fp, predictions)
        output_metadata['predict_fp'] = str(seg_result_fp)

        # Save into HDFStore
        self.hdfstore_output.append(output_table_key, output_data)
        self.hdfstore_output \
            .get_storer(output_table_key) \
            .attrs \
            .metadata = output_metadata

    class Config(WSIBinaryClassifierTest.Config):
        def __init__(self, json_dict: dict, eid: str):
            super().__init__(json_dict, eid)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to config file.')
    parser.add_argument('--eid', type=str, required=True, help='Experiment Identifier')
    args = parser.parse_args()

    json_filepath = args.config_fp
    config = SegmentationTest.Config.load_from_file(
        json_filepath=json_filepath,
        eid=args.eid
    )
    config.parse()
    SegmentationTest(config).run()