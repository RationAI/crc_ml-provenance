from __future__ import annotations

from copy import deepcopy
from typing import NoReturn

from rationai.datagens import Datagen
from rationai.datagens import DataSource
from rationai.generic import StepInterface
from rationai.training.experiments import load_experiment
from rationai.training.models import load_keras_model
from rationai.utils import DirStructure
from rationai.utils import SummaryWriter


class ExperimentRunner(StepInterface):
    """Handles training and evaluation as pipeline steps.
    Initializes and manages a Datagen, a model, and experiment.
    """
    def __init__(self,
                 experiment_class: str,
                 params: dict,
                 dir_struct: DirStructure,
                 summary_writer: SummaryWriter):

        self.dir_struct = dir_struct

        # init train_ds, valid_ds, test_ds
        self._init_datasources(deepcopy(params['data']['dirs']['dataset']))

        # init Experiment
        self.experiment = load_experiment(
            class_name=experiment_class,
            dir_struct=dir_struct,
            summary_writer=summary_writer)

        # init Datagen
        self.experiment.set_datagen(Datagen(deepcopy(params), dir_struct))

        # init Model
        self.experiment.set_model_wrapper(
            load_keras_model(deepcopy(params['model']),
                             ckpt_dir=dir_struct.get('checkpoints')))

    @classmethod
    def from_params(cls: ExperimentRunner,
                    params: dict,
                    self_config: dict,
                    dir_struct: DirStructure,
                    summary_writer: SummaryWriter) -> ExperimentRunner:
        try:
            return cls(experiment_class=self_config['experiment_class'],
                       params=params,
                       dir_struct=dir_struct,
                       summary_writer=summary_writer)
        except ValueError as e:
            print(f'ExperimentRunner init failed: {e}')
            return None

    def train(self, fit_kwargs: dict) -> NoReturn:
        """Initiates the training by calling Experiment's train method

        Keyword arguments:
        fit_kwargs -- a dict with serialized kwargs
                      for keras.Model's fit method
                      (Callbacks get deserialized)
        """
        self.experiment.train(self.train_ds,
                              self.valid_ds,
                              fit_kwargs=fit_kwargs)

    def infer(self, infer_type: str, infer_params: dict, **kwargs) -> NoReturn:
        """Initiates the testing by calling Experiment's infer method

        Keyword arguments:
        infer_type -- testing method type
                      options: "predict", "evaluate", "predict_evaluate"
        infer_params -- a dict with kwargs
                        for keras.Model's predict or evaluate
        """
        self.experiment.infer(self.test_ds,
                              infer_type=infer_type,
                              infer_params=infer_params,
                              **kwargs)

    def _init_datasources(self, dataset_params: dict) -> NoReturn:
        datasource = DataSource(dataset_params,
                                dir_struct=self.dir_struct)
        train_ds, valid_ds, test_ds = datasource.get_train_valid_test()

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
