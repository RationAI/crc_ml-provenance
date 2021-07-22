from copy import deepcopy
from typing import Tuple
from typing import Type

from . import load_augmenter
from . import load_extractor
from .datasources import DataSource
from .generators_tf import (
    KerasBaseGenerator,
    RandomGenerator,
    SequentialGenerator
)
from .samplers import FairSampler
from .samplers import SequentialSampler
from rationai.utils import DirStructure


Generator = Type[KerasBaseGenerator]


class Datagen:
    """Data generator builder class."""

    def __init__(self, config: dict, dir_struct: DirStructure):
        self.config = config
        self.gen_config = config['generator']
        self.dir_struct = dir_struct

    def _get_extractor(self, use_augment: bool):
        """Returns an extractor instance, optionally with an augmenter."""
        identifier = deepcopy(self.gen_config['extractor'])
        identifier['config'] = {'config': self.config,
                                'dir_struct': self.dir_struct,
                                'use_augment': use_augment}
        extractor = load_extractor(identifier)

        if use_augment:
            aug_cfg = self.gen_config['augmenter']
            extractor.set_augmenter(load_augmenter(aug_cfg))

        return extractor

    def get_random_generator(self,
                             steps: int,
                             datasource: Type[DataSource],
                             use_augment: bool = False,
                             valid_type: bool = False) -> Type[RandomGenerator]:
        """Returns a fair random generator.

        Args:
            steps: The number of 'steps' * 'batch_size' learning examples
            are yielded by the genetator after each resampling.

            datasource: A DataSource instance with learning examples.

            use_augment: A bool, whether to perform augmentation.

            valid_type: A bool, whether the generator will be used for validaton
            set. If true, 'steps' will be the total number of yielded examples.

        """
        sampler = FairSampler(datasource,
                              deepcopy(self.gen_config['sampler']))
        extractor = self._get_extractor(use_augment)
        valid_size = steps if valid_type else None
        return RandomGenerator(sampler,
                               extractor,
                               steps,
                               self.gen_config['batch_size'],
                               valid_size=valid_size)

    def get_sequential_generator(
            self,
            datasource: Type[DataSource],
            augment_type: str = 'test') -> Type[SequentialGenerator]:
        """Returns a sequential generator.

        Args:
            datasource: A DataSource instance with learning examples.

            augment_type: A string, uses the corresponding augmentation value
            from the configuration. Options: 'train', 'valid' or 'test'
            Any other value results in use_augment=False.
        """
        sampler = SequentialSampler(datasource,
                                    deepcopy(self.gen_config['sampler']))
        use_augment = self.gen_config.get(f'{augment_type}_augment', False)
        extractor = self._get_extractor(use_augment)
        return SequentialGenerator(sampler, extractor, self.gen_config['batch_size'])

    def get_linear_generator(self,
                             datasource: Type[DataSource],
                             use_augment: bool = False) -> Type[SequentialGenerator]:
        """Returns a linear generator.

        Args:
            datasource: A DataSource instance with learning examples.

            use_augment: A bool, whether to perform augmentation.
        """
        sampler = FairSampler(datasource,
                              deepcopy(self.gen_config['sampler']))
        extractor = self._get_extractor(use_augment)
        return SequentialGenerator(sampler,
                                   extractor,
                                   self.gen_config['batch_size'],
                                   valid_size=self.gen_config['validation_steps'])

    def get_training_generator(
            self,
            train_ds: Type[DataSource],
            valid_ds: Type[DataSource]) -> Tuple[Generator, Generator]:
        """Returns a training and testing generator"""
        # training generator
        train_steps = self.gen_config['steps_per_epoch']
        train_augment = self.gen_config.get('train_augment', False)
        train_gen = self.get_random_generator(steps=train_steps,
                                              datasource=train_ds,
                                              use_augment=train_augment)

        # validation generator
        valid_augment = self.gen_config.get('valid_augment', False)
        if self.gen_config['valid_generator_type'] == 'random':
            valid_gen = self.get_random_generator(
                steps=self.gen_config['validation_steps'],
                datasource=valid_ds,
                use_augment=valid_augment,
                valid_type=True)
        elif self.gen_config['valid_generator_type'] == 'linear':
            valid_gen = self.get_linear_generator(datasource=valid_ds,
                                                  use_augment=valid_augment)

        return train_gen, valid_gen
