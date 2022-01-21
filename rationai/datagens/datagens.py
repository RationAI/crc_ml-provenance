"""Datagen definitions.

Datagens are builders, that are able to compose a data feeding entity
for a particular experiment. Datagen specifies which parts are necessary
to build it. The parameters for these components are then supplied in
a config file.

Generally, the schema for datagens is divided into two parts:
    • definitions - defines classes for each component
    • configs - defines parameters for each component

Examples:

"datagen": {
    "data_sources": {
        "_class": "rationai.datagens.datasources.HDF5DataSource",
        "_data": "/histopat/data/dataset.h5",
        "definitions": {
            "train_valid": {
                "keys": ["train"],
                "names": ["train", "valid"],
                "split_probas": [0.8, 0.2],
                "split_on": null
            }
        }
    },
    "generators": {
        "train_generator": {
            "
            "components": {
                "sampler": "rationai.datagens.samplers.RandomTreeSampler",
                "augmenter": "rationai.datagens.augmenters.NoOpImageAugmenter",
                "extractor": "rationai.datagens.extractors.OpenslideExtractor",
            },
            "config": {
                "sampler": {
                    "epoch_size": 10000,
                    "index_levels": ["label", "slide_name"]
                },
                "augmenter": {},
                "extractor": {
                    "threshold": 0.5
                },
            }
        },
        "valid_generator": {
            "components": {
                "sampler": "rationai.datagens.samplers.SequentialTreeSampler",
                "augmenter": "rationai.datagens.augmenters.NoOpImageAugmenter",
                "extractor": "rationai.datagens.extractors.OpenslideExtractor",
            },
            "config": {
                "sampler": {
                    "index_levels": ["label", "slide_name"]
                },
                "augmenter": {},
                "extractor": {
                    "threshold": 0.5
                },
            }
        }
    }
}

Returns:
    Datagen: Datagen entity
"""

# Standard Imports
from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Type

# Local Imports
from rationai.datagens.augmenters import BaseAugmenter
from rationai.utils.config import ConfigProto
from rationai.datagens.datasources import DataSource
from rationai.datagens.generators import BaseGenerator
from rationai.utils.class_handler import get_class

import logging
log = logging.getLogger('datagens')

class Datagen(ABC):
    """
    TODO: Missing docstring.
    """

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def build_from_template(self):
        """Build generator from template TODO: Generator only? Or anything else potentially?"""


class GeneratorDatagen:
    """
    TODO: Missing docstring.
    """

    def __init__(self, config: GeneratorDatagen.Config):
        self.config = config

    def build_from_template(self):
        """
        TODO: Missing docstring.
        """
        data_sources_dict = self.__build_data_sources_from_template(self.config.data_sources_config)
        generators_dict = self.__build_generators_from_template(self.config.generators_config, data_sources_dict)

        return generators_dict

    def __build_generators_from_template(self, generators_config, data_sources_dict) -> dict[str, BaseGenerator]:
        generators = {}
        for generator_name, generator_config in generators_config.items():
            generator = self.__build_generator_from_template(generator_config, data_sources_dict)
            generators[generator_name] = generator
        return generators

    def __build_generator_from_template(self, generator_config, data_source_dict) -> BaseGenerator:
        definition = generator_config['components']
        components_config = generator_config['configurations']

        data_source = data_source_dict[definition['data_source']]

        sampler_class = get_class(definition['sampler'])
        sampler = self.__build_sampler_from_template(sampler_class, data_source, components_config['sampler'])

        augmenter = None
        if definition['augmenter'] is not None:
            augmenter_class = get_class(definition['augmenter'])
            augmenter = self.__build_augmenter_from_template(augmenter_class, components_config['augmenter'])

        extractor_class = get_class(definition['extractor'])
        extractor = self.__build_extractor_from_template(extractor_class, augmenter, components_config['extractor'])

        generator_class = get_class(definition['generator'])
        return generator_class(sampler, extractor)

    def __build_data_sources_from_template(
            self,
            data_source_configs: dict) -> dict[str, DataSource]:
        # Get DataSource class
        data_source_class = get_class(data_source_configs['_class'])

        # Load dataset path
        dataset_path = Path(data_source_configs['_data'])

        # Construct DataSource from templates
        data_sources = {}
        for _, data_source_config in data_source_configs['definitions'].items():
            data_sources_dict = self.__build_data_source_from_template(
                data_source_class,
                dataset_path,
                data_source_config
            )
            # Merge dictionaries
            data_sources = {**data_sources, **data_sources_dict}
        return data_sources

    @staticmethod
    def __build_data_source_from_template(
            data_source_class: Type[DataSource],
            dataset_path: Path,
            data_source_config_dict: dict) -> dict[str, DataSource]:
        data_source_config = data_source_class.Config(data_source_config_dict)
        data_source_config.parse()
        data_source_dict = data_source_class.load_dataset(
            dataset_fp=dataset_path,
            config=data_source_config
        )

        return data_source_dict

    @staticmethod
    def __build_augmenter_from_template(augmenter_class: Type[BaseAugmenter], augmenter_config_dict: dict):
        augmenter_config = augmenter_class.Config(augmenter_config_dict)
        augmenter_config.parse()
        return augmenter_class(config=augmenter_config)

    @staticmethod
    def __build_sampler_from_template(sampler_class: type, data_source: DataSource, sampler_config_dict: dict):
        sampler_config = sampler_class.Config(sampler_config_dict)
        sampler_config.parse()
        return sampler_class(config=sampler_config, data_source=data_source)

    @staticmethod
    def __build_extractor_from_template(extractor_class: type, augmenter: BaseAugmenter, extractor_config_dict: dict):
        extractor_config = extractor_class.Config(extractor_config_dict)
        extractor_config.parse()
        return extractor_class(config=extractor_config, augmenter=augmenter)

    class Config(ConfigProto):
        """
        TODO: Missing docstring.
        """

        def __init__(self, json_dict=None):
            super().__init__(json_dict)

            self.data_sources_config = None
            self.generators_config = None

        def parse(self):
            self.data_sources_config = self.config['data_sources']
            self.generators_config = self.config['generators']
