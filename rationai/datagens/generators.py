"""
TODO: Missing docstring.
"""
import logging
import hashlib
from time import time
from typing import List, Tuple
from typing import NoReturn

import numpy as np
from tensorflow.keras.utils import Sequence
from torch.utils.data import Dataset

from rationai.datagens.extractors import Extractor
from rationai.datagens.samplers import SampledEntry
from rationai.datagens.samplers import TreeSampler
from rationai.utils.utils import divide_round_up
from rationai.utils.config import ConfigProto
from rationai.utils.provenance import SummaryWriter


log = logging.getLogger('generators')

sw_log = SummaryWriter.getLogger('provenance')


class BaseGenerator:
    """
    Base class for data generators.

    Attributes
    ----------
    sampler : TreeSampler
        Implementation of the sampling logic.
    extractor : Extractor
        Handles extraction of the samples from actual data.
    epoch_samples : pandas.DataFrame
        The sampled data points of the currently generated epoch.
    """

    def __init__(self, config: ConfigProto, name: str, sampler: TreeSampler, extractor: Extractor):
        self.name = name
        self.config = config
        self.sampler = sampler
        self.extractor = extractor
        self.epoch_samples: list[SampledEntry] = []

    def _generate_samples(self) -> List[SampledEntry]:
        """Get a sampled epoch as a pandas dataframe.

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

    def __init__(self, config: ConfigProto, name: str, sampler: TreeSampler, extractor: Extractor):
        super().__init__(config, name, sampler, extractor)

        self.epoch_samples = self._generate_samples()
        self.batch_size = self.config.batch_size
        sw_log.set('iters', sw_log.vars['gen_counter'], self.name, 'sha256', value=self.get_epoch_samples_digest())

    def set_batch_size(self, batch_size: int):
        """
        TODO: Missing docstring.
        """
        self.batch_size = batch_size

    def __len__(self) -> int:
        return divide_round_up(len(self.epoch_samples), self.batch_size)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
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
        """
        TODO: Missing docstring.
        """
        t0 = time()
        # TODO: Decide how to rework this.
        if self.config.resample:
            self.epoch_samples = self.sampler.on_epoch_end()
            log.info(f'Keras generator resampled on epoch end ({int(time() - t0)}s)')

        sw_log.set('iters', sw_log.vars['gen_counter'], self.name, 'sha256', value=self.get_epoch_samples_digest())

    def get_epoch_samples_digest(self):
        """For now extremely naive solution for digest for provenance sample usecase."""
        s = str(self.epoch_samples)
        return hashlib.sha256(s.encode('UTF-8')).hexdigest()

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.batch_size = None
            self.resample = None

        def parse(self):
            self.batch_size = self.config['batch_size']
            self.resample = self.config['resample']

class BaseGeneratorPytorch(BaseGenerator, Dataset):
    """Base class for generators based on torch.utils.data.Dataset

    Implements the interface between PyTorch and the custom sampling & extraction.
    """

    def __init__(self, config: ConfigProto, name: str, sampler: TreeSampler, extractor: Extractor):
        super().__init__(config, name, sampler, extractor)

    def __len__(self) -> int:
        return len(self.epoch_samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
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
        """
        TODO: Missing docstring.
        """
        t0 = time()
        # TODO: Decide how to rework this.
        self.epoch_samples = self.sampler.on_epoch_end()
        log.info(f'PyTorch generator resampled on epoch end ({int(time() - t0)}s)')

    class Config(ConfigProto):
        def __init__(self, json_dict: dict):
            super().__init__(json_dict)
            self.resample = None

        def parse(self):
            self.resample = self.config['resample']
