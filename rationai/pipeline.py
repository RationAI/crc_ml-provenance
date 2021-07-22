import argparse
import json
import logging
import numpy as np
import os
import tensorflow as tf
from copy import deepcopy
from pathlib import Path

from .utils import SummaryWriter
from .generic import ExperimentInitializer
from .generic import StepExecutor

log = logging.getLogger('runner')
tf_log = logging.getLogger('tensorflow').setLevel(logging.ERROR)   # Suppress annoying deprecation warnings
log.info(f'TensorFlow Version: {tf.__version__}')


def main(params: dict, description: str, device: str):

    # Set seed
    seed = get_seed(params)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    log.info(f'Seed: {seed}')
    log.info(f'Experiment Description: {description}')
    device = f'GPU {device}' if int(device) >= 0 else 'CPU'
    log.info(f'Device: {device}')

    # Get Job ID if this is METACENTRUM
    if 'PBS_JOBID' in os.environ:
        log.info(f"Computation on metacentrum. Job ID: {os.environ['PBS_JOBID']}")
        params['job_id'] = os.environ['PBS_JOBID']

    # Initialize summary writer
    log.info('Initialising SummaryWriter')
    sw = SummaryWriter(params, description)
    sw.set_value('device', value=device)

    # Initialize experiment
    log.info('Initializing experiment.')
    exp_init = ExperimentInitializer(
        exp_params=deepcopy(params['experiment']),
        data_dirs_params=deepcopy(params['data']['dirs']))

    # Path managing
    dir_struct = exp_init.get_dir_struct()
    sw.set_path(dir_struct.get('expdir') / 'summary.json')
    dir_struct.log_paths_to_summary(sw)

    # Execute the pipeline steps
    StepExecutor(
        ordered_steps=deepcopy(params['experiment'].get('ordered_steps', [])),
        step_defs=deepcopy(params.get('step_definitions', dict())),
        dir_struct=dir_struct,
        summary_writer=sw,
        params=deepcopy(params)
    ).run_all()

    log.info(f'Experiment ID: {exp_init.eid}')
    dir_struct.log_paths_to_summary(sw)


def load_params(params_filepath: Path) -> dict:
    with params_filepath.open('r') as params_f:
        params = json.load(params_f)
        return params


def get_seed(params: dict, low=0, high=1000000) -> int:
    """Retrieves a PRNG seed from params or generates a new one.

    The new seed gets logged by ExperimentInitializer class
    with the rest of the config.
    """
    if 'seed' in params['experiment']:
        return params['experiment']['seed']

    # The new seed will get logged later
    seed = np.random.randint(low, high)
    params['experiment']['seed'] = seed
    return seed


if __name__ == '__main__':
    """
    The pipeline runner.

    USAGE:

    $ python3 -m rationai.pipeline CONFIG_FILE [--gpu GPU_ID] [--desc DESCRIPTION]

        CONFIG_FILE: Path to JSON configuration file.
        GPU_ID: ID of a GPU to use; alternatively -1 for CPU (default: 0).
        DESCRIPTION: Run description differentiates running processes.
    """

    parser = argparse.ArgumentParser()

    # Config params
    parser.add_argument(type=Path,
                        dest='config',
                        help='Path to a JSON configuration file')

    parser.add_argument('--gpu',
                        type=str,
                        required=False,
                        default='0',
                        help='GPU ID or -1 for CPU (Default 0)')

    parser.add_argument('--desc',
                        dest='description',
                        type=str,
                        required=False,
                        default='No description provided',
                        help='Experiment description')

    args = parser.parse_args()

    # Do not manipulate the variable when using the default value
    # (because of Metacentrum runs).
    # Should only be used for convenience when running on "our" servers.
    if args.gpu != 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(load_params(args.config), args.description, args.gpu)
