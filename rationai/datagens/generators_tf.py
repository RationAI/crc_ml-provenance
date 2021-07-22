import logging
import pandas as pd
from tensorflow.keras.utils import Sequence
from time import time
from nptyping import NDArray
from typing import (
    NoReturn,
    Optional,
    Tuple,
    Type
)

from rationai.datagens.extractors import SlideExtractor
from rationai.datagens.samplers import BaseSampler
from rationai.utils import divide_round_up

log = logging.getLogger('generator')


class KerasBaseGenerator(Sequence):
    """Base class for generators based on tf.keras.utils.Sequence

    Implements the interface between Keras and
    the custom sampling & extraction.
    """
    def __init__(self,
                 sampler: Type[BaseSampler],
                 extractor: Type[SlideExtractor],
                 batch_size: int):
        """
        Args:
            sampler: BaseSampler subclass
                Implements the sampling logic.

            extractor: SlideExtractor subclass
                Extracts the samples.

            steps_per_epoch: int
                In each step `batch_size` batches are processed

            batch_size: int
                Learning batch size (default 1)
        """
        self.sampler = sampler
        self.extractor = extractor
        self.batch_size = batch_size
        self.current_samples = []

    def __getitem__(self, index: int) -> Tuple[NDArray, NDArray]:
        """Returns index-th learning batch"""
        samples = self.current_samples[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.extractor(samples)
        return X, y

    def _generate_samples(self, num_samples: int) -> pd.DataFrame:
        """Returns a sampled epoch as a pd.DataFrame"""
        return pd.concat([self.sampler.sample() for _ in range(num_samples)])


class RandomGenerator(KerasBaseGenerator):
    """Generators a new set of samples for each epoch"""
    def __init__(self,
                 sampler: Type[BaseSampler],
                 extractor: Type[SlideExtractor],
                 steps_per_epoch: int,
                 batch_size: int = 1,
                 valid_size: Optional[int] = None):
        """
        Args:
            sampler: BaseSampler subclass
                Implements the sampling logic.

            extractor: SlideExtractor subclass
                Extracts the samples.

            steps_per_epoch: int
                In each step `batch_size` batches are processed.

            batch_size: int
                Learning batch size. (default 1)

            valid_size: float / int
                A total number of learning examples to generate.
                (overrides `steps_per_epoch` when used as validation generator)
        """
        super().__init__(sampler, extractor, batch_size)
        self.steps_per_epoch = steps_per_epoch
        self.valid_size = valid_size
        samples_num = valid_size if valid_size else self.batch_size * self.steps_per_epoch
        self.current_samples = self._generate_samples(samples_num)

    def __len__(self) -> int:
        return self.steps_per_epoch \
            if not self.valid_size \
            else divide_round_up(len(self.current_samples), self.batch_size)

    def on_epoch_end(self) -> NoReturn:
        t0 = time()
        samples_num = self.valid_size \
            if self.valid_size else self.batch_size * self.steps_per_epoch
        self.current_samples = self._generate_samples(samples_num)
        log.info(f'Random generator resampled on epoch end ({int(time()-t0)}s)')


class SequentialGenerator(KerasBaseGenerator):
    """Generates learning examples linearly, one sequence at a time.
    """
    def __init__(self,
                 sampler: Type[BaseSampler],
                 extractor: Type[SlideExtractor],
                 batch_size: int = 1,
                 valid_size: Optional[int] = None):
        """
        Args:
            sampler: BaseSampler subclass
                Implements the sampling logic.

            extractor: SlideExtractor subclass
                Extracts the samples.

            steps_per_epoch: int
                In each step `batch_size` batches are processed

            batch_size: int
                Learning batch size (default 1)

            valid_size: float / int
                A total number of learning examples to generate
                (when used as validation generator)
        """
        super().__init__(sampler, extractor, batch_size)
        self.valid_size = valid_size
        samples_num = self.valid_size if self.valid_size else len(self.sampler)
        self.current_samples = self._generate_samples(samples_num)
        self.len = divide_round_up(len(self.current_samples), self.batch_size)

    def __len__(self) -> int:
        """Returns the length of the current sequence."""
        return self.len

    def num_of_sequences(self) -> Optional[int]:
        """Returns number of sequences
        or None if its sampler is not seqeuential.
        """
        if hasattr(self.sampler, 'sequences'):
            return len(self.sampler.sequences)
        log.info('SequentialSampler does not have attribute `sequences`')
        return None

    def prepare_sequence(self, idx: int) -> str:
        """Prepares the idx-th data sequence (e.g. idx-th WSI slide)
        and returns the id of the current sequence.

        Side effect: changes length of a generator accordingly.
        """
        sequence_id = self.sampler.prepare_sequence(idx)
        samples_num = self.valid_size if self.valid_size else len(self.sampler)
        self.current_samples = self._generate_samples(samples_num)
        self.len = divide_round_up(len(self.current_samples), self.batch_size)
        return sequence_id
