import abc
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from nptyping import NDArray
from typing import NoReturn

from rationai.datagens import Datagen
from rationai.datagens import DataSource
from rationai.training.callbacks import load_keras_callbacks
from rationai.training.models import BaseModel
from rationai.utils import (
    DirStructure,
    join_module_path,
    load_from_module,
    Mode,
    SummaryWriter,
)

log = logging.getLogger('experiments')


def load_experiment(class_name: str, *args, **kwargs):
    """Loads experiment class defined by class_name in identifier"""
    path = join_module_path(__name__, class_name)
    return load_from_module(path, *args, **kwargs)


class ExperimentInterface(metaclass=abc.ABCMeta):
    """Interface for Experiment classes"""

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'infer') and
                callable(subclass.infer) or
                NotImplemented)

    @abc.abstractmethod
    def set_model_wrapper(self, model_wrapper: BaseModel) -> NoReturn:
        """Assigns a model wrapper to the experiment"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_datagen(self, datagen: Datagen) -> NoReturn:
        """Assigns an experiment to the experiment"""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self,
              train_ds: DataSource,
              valid_ds: DataSource,
              train_params: dict) -> NoReturn:
        """Runs training.
        Uses train and validation DataSource and train parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self,
              test_ds: DataSource,
              infer_params: dict) -> NoReturn:
        """Runs inference.
        Uses test DataSource and infer parameters."""
        raise NotImplementedError


class GenericExperiment(ExperimentInterface):
    """GenericExperiment implements generic training"""

    def __init__(self,
                 dir_struct: DirStructure,
                 summary_writer: SummaryWriter):

        self.summary_writer = summary_writer
        self.dir_struct = dir_struct

        # Need to be set later
        self.model_wrapper = None
        self.datagen = None

        self.summary_writer.set_value('experiment_start', value=SummaryWriter.now())

    def set_model_wrapper(self, model_wrapper: BaseModel) -> NoReturn:
        self.model_wrapper = model_wrapper

    def set_datagen(self, datagen: Datagen) -> NoReturn:
        self.datagen = datagen

    def deserialize_callbacks(self, params: dict) -> dict:
        """Deserializes callbacks inside params and returns kwargs
        ready for model's fit/predict/infer method."""
        kwargs = deepcopy(params)

        if 'callbacks' in params:
            kwargs['callbacks'] = \
                load_keras_callbacks(
                    callbacks=params['callbacks'],
                    out_dir=self.dir_struct.get('expdir') / 'callbacks')
        return kwargs

    def train(self,
              train_ds: DataSource,
              valid_ds: DataSource,
              fit_kwargs: dict) -> NoReturn:
        """Generic method for a model to fit a generator & log metrics"""
        self.fit_kwargs = deepcopy(fit_kwargs)

        _keys = [Mode.Train.value, 'start_time']
        self.summary_writer.set_value(*_keys, value=SummaryWriter.now())

        # Get train & valid generators
        train_gen, valid_gen = self.datagen.get_training_generator(
            train_ds=train_ds,
            valid_ds=valid_ds)

        # Fit generator
        train_log = self.model_wrapper.model.fit(
            x=train_gen,
            validation_data=valid_gen,
            **self.deserialize_callbacks(fit_kwargs))

        # Log metrics
        for metric_name, metric_values in train_log.history.items():
            if metric_name.startswith('val'):
                mode = Mode.Validate.value
                metric_name = metric_name[4:]
            else:
                mode = Mode.Train.value

            for epoch, metric_value in enumerate(metric_values):
                _keys = [mode, 'summary', f'epoch_{epoch}', 'metrics', metric_name]
                self.summary_writer.set_value(*_keys, value=f'{metric_value:.5f}')

        # Log time
        self.summary_writer.set_value(Mode.Train.value,
                                      'end_time',
                                      value=SummaryWriter.now())
        self.summary_writer.update_log()

    @abc.abstractmethod
    def save_preditions(self,
                        results: NDArray,
                        sequence_name: str,
                        pddf: pd.DataFrame) -> NoReturn:
        """Should handle prediction saving for the specific data format"""
        raise NotImplementedError('Implement save_predictions method in your subclass')

    def _prepare_infer_dir_structure(self, infer_type: str):
        if infer_type != 'evaluate':
            path = self.dir_struct.get('expdir')
            self.dir_struct.add('predictions', path / 'predictions', create=True)

    def infer(self,
              test_ds: DataSource,
              infer_type: str,
              infer_params: dict,
              max_sequences: int = None) -> NoReturn:
        """A generic method for model validation and prediction making.
        For saving predictions, abstract method 'save_preditions' has to be implemented.

        Args:
        test_ds : DataSource
            test subset

        infer_type: str
            Method type to use for inference:
                'predict' - saves predictions
                'evaluate' - evaluates metrics
                'predict_evaluate' - saves predictions and evaluates metrics

        infer_params: dict
            Keyword arguments passed to the used method.
            For 'predict_evaluate' use evaluate's signature.

        max_sequences: int
            Limits the number of processed sequences.
            (default None - only for debugging)
        """
        self.infer_params = deepcopy(infer_params)

        # Init kwargs (serialize callbacks)
        infer_kwargs = self.deserialize_callbacks(self.infer_params)

        # Create dir
        self._prepare_infer_dir_structure(infer_type)

        if infer_type == 'predict_evaluate':
            # Performs predict & evaluate in one test_step
            self.model_wrapper.override_predict_evaluate()

        if infer_type in ['evaluate', 'predict_evaluate']:
            if 'return_dict' not in infer_kwargs:
                infer_kwargs['return_dict'] = True

        # Prepare model
        self.model_wrapper.test_mode()

        # Get generator
        test_generator = self.datagen.\
            get_sequential_generator(test_ds, augment_type='test')

        y_pred = None  # predictions
        metrics = None  # metric logs

        # TEST EACH SEQUENCE SEPARATELY
        num_of_sequences = test_generator.num_of_sequences()

        if max_sequences is not None:
            num_of_sequences = min(max_sequences, num_of_sequences)

        for sequence_idx in range(num_of_sequences):
            # Prepare sequence
            sequence_name = test_generator.prepare_sequence(sequence_idx)
            log.info(f'Infering: {sequence_name} [{sequence_idx+1}/{num_of_sequences}]')

            # Log time
            _keys = [Mode.Test.value, 'summary', sequence_name, 'start_time']
            self.summary_writer.set_value(*_keys, value=SummaryWriter.now())

            log.debug(f'Computing predictions for {sequence_name}')

            if infer_type == 'evaluate':
                metrics = self.model_wrapper.model.evaluate(x=test_generator,
                                                            **infer_kwargs)
            elif infer_type == 'predict_evaluate':
                result = self.model_wrapper.model.evaluate(x=test_generator,
                                                           **infer_kwargs)
                y_pred, metrics = result['y_preds'], result['logs']
            elif infer_type == 'predict':
                y_pred = self.model_wrapper.model.predict(x=test_generator,
                                                          **infer_kwargs)
            else:
                raise ValueError(
                    f'Unknown inference method type: {infer_type}\n'
                    'Valid types are: "evaluate", "predict", "predict_evaluate"')

            if y_pred is not None:
                # Save predictions
                self.save_preditions(y_pred,
                                     sequence_name=sequence_name,
                                     pddf=test_generator.sampler.sequence_df)

            _keys = [Mode.Test.value, 'summary', sequence_name]
            if metrics is not None:
                # Log sequence metrics
                seq_metrics = {k: v for k, v in metrics.items()
                               if not k.startswith('total-')}
                self.summary_writer.set_value(
                    *_keys, 'metrics', value=seq_metrics)

            # Log end time
            self.summary_writer.set_value(
                *_keys, 'end_time', value=SummaryWriter.now())
            self.summary_writer.update_log()

        # Log total metrics
        total_metrics = self.model_wrapper.get_total_metrics()
        if total_metrics:
            log.debug('Logging total metrics')
            results = dict((tm._name, float(tm.result().numpy()))
                           for tm in total_metrics)

            _keys = [Mode.Test.value, 'summary', 'total_metrics']
            self.summary_writer.set_value(*_keys, value=results)

        self.summary_writer.update_log()


class ClassificationExperiment(GenericExperiment):
    """Implements save_predictions for classification problem.
    Predicted class is put inside a csv dataframe file.

    E.g. prostate cancer detection where model clasifies input
    either as negative or positive (carcinoma)"""
    def __init__(self,
                 dir_struct: DirStructure,
                 summary_writer: SummaryWriter):
        super().__init__(dir_struct=dir_struct, summary_writer=summary_writer)

    def save_preditions(self,
                        results: NDArray,
                        sequence_name: str,
                        pddf: pd.DataFrame) -> NoReturn:
        """Saves predictions to disk"""
        log.debug('Writing predictions to disk')

        fp = self.dir_struct.get('predictions') / sequence_name
        pddf['predict'] = results.ravel().astype(np.float64)
        pddf.to_csv(str(fp), sep=';', float_format='%.4f')

        _keys = [Mode.Test.value, 'summary', Path(fp).stem, 'results']
        self.summary_writer.set_value(*_keys, value=str(fp))


class SegmentationExperiment(GenericExperiment):
    """Implements inference for tissue segmentation,
    where an image is a result of prediction"""
    def __init__(self,
                 dir_struct: DirStructure,
                 summary_writer: SummaryWriter):
        super().__init__(dir_struct=dir_struct, summary_writer=summary_writer)

    def save_preditions(self,
                        results: NDArray,
                        sequence_name: str,
                        pddf: pd.DataFrame) -> NoReturn:
        """Saves the segmentation predictions as images"""
        # TODO: Implement saving of results (images) for cytoseg segmentation.
        # However, at the moment there is no need to save the results,
        # because after evaluation they are not further utilized.
        pass
