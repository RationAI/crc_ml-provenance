# Standard Imports
from pathlib import Path
import argparse
import logging
import shutil
log = logging.getLogger('ex_eval')

# Third-party Imports

# Local Imports
from rationai.training.base.experiments import Experiment
from rationai.utils.class_handler import get_class
from rationai.utils.provenance import SummaryWriter

sw_log = SummaryWriter.getLogger('provenance')

class WSIBinaryClassifierEval(Experiment):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.setup()
        self.eval()

    def eval(self):
        eval_gen = self.generators_dict[self.config.eval_gen]

        while eval_gen.sampler.active_node is not None:
            log.info(f'Evaluating node: {eval_gen.sampler.active_node.node_name}')
            for input_dict in eval_gen:
                for evaluator in self.evaluators:
                    evaluator.update_state(input_dict)

            table_key = eval_gen.epoch_samples[0].entry['_table_key']
            hash = SummaryWriter.hash_table(eval_gen.sampler.data_source.source, table_key)
            for evaluator in self.evaluators:
                log.info(f'{evaluator.name}: {evaluator.result()}')
                sw_log.set('eval', hash, f'{evaluator.name}', value=float(evaluator.result()))
                evaluator.reset_state()

            eval_gen.sampler.next()
            eval_gen.on_epoch_end()
            sw_log.vars['gen_counter'] += 1

    def setup(self):
        # Build Datagen
        datagen_config = self.config.datagen_class.Config(
            self.config.datagen_config
        )
        datagen_config.parse()
        self.generators_dict = self.config.datagen_class(datagen_config) \
            .build_from_template()

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
        def __init__(self, json_dict: dict, eid: str):
            super().__init__(json_dict, eid)

            # Generator Selection
            self.eval_gen = None

            # Datagen Configuration
            self.datagen_class = None
            self.datagen_config = None

        def parse(self):
            super().parse()

            definitions_config = self.config['definitions']
            configurations_config = self.config.get('configurations', dict())

            # Generator Selection
            self.eval_gen = self.config.get('eval_generator')

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
    config = WSIBinaryClassifierEval.Config.load_from_file(
        json_filepath=json_filepath,
        eid=args.eid
    )
    config.parse()
    WSIBinaryClassifierEval(config).run()

    # Copy configuration file
    shutil.copy2(args.config_fp, Experiment.Config.experiment_dir / args.config_fp.name)
    sw_log.set('config_file', value=str(Path(Experiment.Config.experiment_dir / args.config_fp.name).resolve()))
    sw_log.to_json(Experiment.Config.experiment_dir / 'prov_eval.log')
