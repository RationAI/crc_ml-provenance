import pandas as pd
import numpy as np
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import (
    List,
    Tuple,
    Type
)

from .datasources import DataSource

log = logging.getLogger('sampler')


class BaseSampler(ABC):
    """Base class for sampling logic implementation."""
    def __init__(self, datasource: Type[DataSource]):
        self.datasource = datasource
        self.sampling_cols = []

    @abstractmethod
    def sample(self) -> Type[pd.DataFrame]:
        """Method samples a learning example representation."""
        raise NotImplementedError('sample() method not implemented')

    def _get_sources(self) -> List[Tuple[List[str], Type[pd.DataFrame]]]:
        """Returns a list of pairs: ( [sampling_strat], respective sub_df )"""

        if self.datasource.is_composed:
            compression = 'gzip' if Path(self.datasource.source[0]).suffix == '.gz' else None
            sources = [([], pd.read_pickle(df_path, compression=compression))
                       for df_path in self.datasource.source]
        else:
            sources = [([], self.datasource.source)]

        for col in self.sampling_cols:
            sub_sources = []

            # divide each DF by unique values in column['col']
            for (strat, df) in sources:
                df_unique_vals = df[col].unique()
                # NOTE: add directly to sampler dict instead of storing 'sources'
                # extend the sources with new sub_dfs (each paired with a "sampl. tree path")
                sub_sources.extend((strat + [uval], df.loc[df[col] == uval])
                                   for uval in df_unique_vals)

            sources = sub_sources

        return sources

    def _build_sampling_tree(self) -> dict:
        """Builds a sampling tree dictionary used for sampling.

        All paths in the dictionary lead to existing non-empty dataframes."""
        sources = self._get_sources()

        sampler = dict()

        # NOTE: strat has to be defined, otherwise the sampling is highly inefficient
        for strat, sub_df in sources:
            s = sampler
            for i, col in enumerate(strat):
                if i == len(strat) - 1:
                    s[col] = sub_df if col not in s else pd.concat([s[col], sub_df])
                else:
                    if col in s:
                        s = s[col]
                    else:
                        s[col] = dict()
                        s = s[col]
        return sampler


class FairSampler(BaseSampler):
    """
    Uses a sampling strategy defined in the config file for sampling.

    example: { 'label':[], 'slide': []}
    1. choose random label L
        - random unique value from column `label`
    2. choose random slide S (having label L)
        - random unique value from column `slide`
    3. sample a random row of the given sub dataframe
        - all rows have label L and belong to slide S

    The arrays can be used to define sampling probabilities
    at each level in ascending order.
    """

    def __init__(self, datasource: Type[DataSource], sampler_config: dict):
        super().__init__(datasource)
        # always define a strat when working with mutliple sources, otherwise it would be highly ineffective
        self.sampling_strat = sampler_config['sampling_strat']
        self.sampling_cols = list(self.sampling_strat.keys())
        self.sampler = self._get_sampler()

    def _get_sampler(self):
        """Builds a sampling tree based on the sampling_strat or returns a data frame"""
        log.debug('Building fair sampler')
        if not self.datasource.is_composed and not self.sampling_strat:
            return self.datasource.source
        return self._build_sampling_tree()

    def sample(self):
        if not self.sampling_strat:
            # should not be executed with multiple sources (inefficient)
            row_idx = np.random.randint(low=0, high=len(self.datasource.source))
            return self.datasource.source.iloc[[row_idx]]

        sub_sampler = self.sampler

        for col in self.sampling_cols:
            probabilities = self.sampling_strat[col] if self.sampling_strat[col] else None
            choice = np.random.choice(sorted(sub_sampler.keys()), p=probabilities)
            sub_sampler = sub_sampler[choice]

        row_idx = np.random.randint(low=0, high=len(sub_sampler))
        return sub_sampler.iloc[[row_idx]]


class SequentialSampler(BaseSampler):
    """
    Linearly samples from one sequence (e.g., WSI) at a time.

    Sampler config needs to contain `sequence_col` to designate
    which data frame column to use as a sequence.
    """
    def __init__(self, datasource: Type[DataSource], sampler_config: dict):
        super().__init__(datasource)
        # NOTE: if there is no sequence column -> linear sampler
        # -> concat all or linear walk through dfs?
        self.sequence_column = sampler_config['sequence_column']
        self.sampler = self._get_sampler()
        self.prepare_sequence(0)

    def _get_sampler(self) -> dict:
        """Returns a built sampling tree"""
        log.debug('Building sequential sampler')
        # Groups data into sequences by 'sequence_col' of coord_maps
        sampling_strat = {'sampling_strat': {self.sequence_column: []}}
        sampler = FairSampler(self.datasource, sampling_strat).sampler
        self.sequences = list(sampler.keys())
        return sampler

    def __len__(self):
        """Returns the length of the current sequence."""
        return self.len

    def sample(self) -> Type[pd.DataFrame]:
        """Returns the next sample"""
        return next(self.gen)[1].to_frame().transpose()

    def prepare_sequence(self, idx: int):
        """Prepares next sequence and returns its id - e.g. WSI filename"""
        self.sequence_df = self.sampler[self.sequences[idx]]
        self.len = len(self.sequence_df)
        self.gen = self.sequence_df.iterrows()
        log.debug(f'Changed sequence to: {self.sequences[idx]}')
        return self.sequences[idx]
