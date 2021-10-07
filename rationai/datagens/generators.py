import logging
from time import time
from typing import Type, Tuple, Any, Optional, NoReturn

import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence
from torch.utils.data import Dataset


log = logging.getLogger('generators')


class BaseGenerator:
    """
    Base class for data generators.

    # TODO: Add proper type annotations when ready

    Attributes
    ----------
    sampler : 'SamplerBase'
        Implementation of the sampling logic.
    extractor : 'ExtractorBase'
        Handles extraction of the samples from actual data.
    batch_size : int
        The number of samples in extracted batches.
    epoch_samples : pandas.DataFrame
        The sampled data points of the currently generated epoch.
    """

    # TODO: Add proper type annotations when ready
    def __init__(self, sampler: 'SamplerBase', extractor: 'ExtractorBase'):
        self.sampler = sampler
        self.extractor = extractor
        self.epoch_samples: list['SampledEntry'] = []

    # TODO: Connect annotations when ready.
    def _generate_samples(self) -> list['SampledEntry']:
        """Get a sampled epoch as a pandas dataframe.

        Parameters
        ----------
        num_samples : int
            Number of samples in the epoch.

        Return
        ------
        pandas.DataFrame
            Sampled epoch.
        """
        return self.sampler.sample()


class BaseGeneratorKeras(BaseGenerator, Sequence):
    """Base class for generators based on tf.keras.utils.Sequence

    Implements the interface between Keras and the custom sampling & extraction.
    """

    def __init__(self, sampler: 'SamplerBase', extractor: 'ExtractorBase'):
        super().__init__(sampler, extractor)
        self.batch_size = None
        self.epoch_samples = self._generate_samples()

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def __len__(self) -> int:
        # TODO: Add divide_round_up
        return divide_round_up(len(self.epoch_samples), self.batch_size)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get data batch at `index` from `self.epoch_samples`.

        Parameters
        ----------
        index : int
            The position of the batch.

        Return
        ------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple representing a batch with the format (input_data, label_data).
        """
        return self.extractor(self.epoch_samples[index * self.batch_size:(index + 1) * self.batch_size])

    def on_epoch_end(self) -> NoReturn:
        t0 = time()
        self.epoch_samples = self.sampler.on_epoch_end()
        log.info(f'Keras generator resampled on epoch end ({int(time() - t0)}s)')


class BaseGeneratorPytorch(BaseGenerator, Dataset):
    """Base class for generators based on torch.utils.data.Dataset

    Implements the interface between PyTorch and the custom sampling & extraction.
    """

    def __init__(self, sampler: 'SamplerBase', extractor: 'ExtractorBase'):
        super().__init__(sampler, extractor)

    def __len__(self) -> int:
        return len(self.epoch_samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get data batch at `index` from `self.epoch_samples`.

        Parameters
        ----------
        index : int
            The position of the batch.

        Return
        ------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple representing a batch with the format (input_data, label_data).
        """
        return self.extractor(self.epoch_samples[index])

    def on_epoch_end(self) -> NoReturn:
        t0 = time()
        self.epoch_samples = self.sampler.on_epoch_end()
        log.info(f'PyTorch generator resampled on epoch end ({int(time() - t0)}s)')
