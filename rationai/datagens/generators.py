"""
TODO: Missing docstring.
"""
import logging
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


log = logging.getLogger('generators')

#! DELETE THIS
handler = logging.FileHandler('/mnt/data/home/matejg/Project/prov-wittner/prov.log')
prov_log = logging.getLogger('prov-log')
prov_log.setLevel(logging.INFO)
prov_log.addHandler(handler)


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

    def __init__(self, sampler: TreeSampler, extractor: Extractor):
        self.name = None
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

    def __init__(self, sampler: TreeSampler, extractor: Extractor):
        super().__init__(sampler, extractor)
        self.batch_size = None
        self.epoch_samples = self._generate_samples()
        if self.epoch_samples is not None:
            prov_log.info(f'{self.name} epoch samples hash: {sum([hash(frozenset(x.entry.items())) for x in self.epoch_samples])}')
        self.resample = None    # TODO: Move to config
        self.name = None # TODO: Move to config

    def set_batch_size(self, batch_size: int):
        """
        TODO: Missing docstring.
        """
        self.batch_size = batch_size

    def __len__(self) -> int:
        # TODO: Add divide_round_up
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
        if self.resample:
            self.epoch_samples = self.sampler.on_epoch_end()
            log.info(f'Keras generator resampled on epoch end ({int(time() - t0)}s)')
            if self.epoch_samples is not None:
                prov_log.info(f'{self.name} epoch samples hash: {sum([hash(frozenset(x.entry.items())) for x in self.epoch_samples])}')


class BaseGeneratorPytorch(BaseGenerator, Dataset):
    """Base class for generators based on torch.utils.data.Dataset

    Implements the interface between PyTorch and the custom sampling & extraction.
    """

    def __init__(self, sampler: TreeSampler, extractor: Extractor):
        super().__init__(sampler, extractor)

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
