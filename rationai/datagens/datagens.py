# Standard Imports
from abc import ABC
from abc import abstractmethod
from pathlib import Path

# Third-party Imports

# Local Imports
from rationai.datagens.datagens_config import DatagenConfig
from rationai.datagens.datasources import DataSource
from rationai.datagens.generators import Generator
from rationai.utils.class_handler import get_class


class Datagen(ABC):

    @abstractmethod
    def __init__(self):
        """Class Constructor"""

    @abstractmethod
    def __build_from_template(self):
        """Build generator from template"""

class GeneratorDatagen:

    def __init__(self, config: DatagenConfig):
        self.config = config

    def build_from_template(self):
        data_sources_cfg = self.config['data_sources']
        data_sources_dict = self.__build_data_sources_from_template(data_sources_cfg)

        generators_cfg = self.config['generators']
        generators_dict = self.__build_generators_from_template(generators_cfg, data_sources_dict)

        return generators_dict

    def __build_generators_from_template(self, generators_config, data_sources_dict) -> dict[str, Generator]:
        generators = {}
        for generator_name, generator_config in generators_config.items():
            generator = self.__build_generator_from_template(generator_config, data_sources_dict)
            generators[generator_name] = generator
        return generators

    def __build_generator_from_template(self, generator_config, data_source_dict) -> Generator:
        definition = generator_config['definition']
        sampler_class = get_class(definition['sampler'])
        augmenter_class = get_class(definition['augmenter'])
        extractor_class = get_class(definition['extractor'])
        generator_class = get_class(definition['generator'])

        data_source = data_source_dict[definition['data_source']]

        components_config = generator_config['configuration']
        sampler = self.__build_sampler_from_template(sampler_class, data_source, components_config['sampler'])
        augmenter = self.__build_augmenter_from_template(augmenter_class, components_config['augmenter'])
        extractor = self.__build_extractor_from_template(extractor_class, augmenter, components_config['extractor'])

        return generator_class(sampler, extractor)

    def __build_data_sources_from_template(
        self,
        data_source_configs: dict) -> dict[str, DataSource]:
        # Get DataSource class
        data_source_class = get_class(
            'rationai.datagens.datasources',
            data_source_configs['_class']
        )

        # Load dataset path
        dataset_path = data_source_configs['_data']

        # Construct DataSource from teplates
        data_sources = {}
        for _, data_source_config in data_source_configs.items():
            data_sources_dict = self.__build_data_source_from_template(
                data_source_class,
                dataset_path,
                data_source_config
            )
            data_sources = data_sources | data_sources_dict
        return data_sources

    def __build_data_source_from_template(
        self,
        data_source_class: type,
        dataset_path: Path,
        data_source_config: dict) -> dict[str, DataSource]:
        data_source = data_source_class.load_dataset(
            dataset_fp=dataset_path,
            keys=data_source_config['keys']
        )

        if len(data_source_config['names']) == 1:
            return {data_source_config['names'][0]: data_source}

        data_sources = data_source.split(
            sizes=data_source_config['split_probas'],
            key=data_source_config['split_on']
        )

        return dict(zip(data_source_config['names'], data_sources))

    def __build_augmenter_from_template(self):
        # TODO: Once Augmenter class is known.
        pass

    def __build_sampler_from_template(self, sampler_class: type, sampler_config: dict):
        return sampler_class(**sampler_config)

    def __build_extractor_from_template(self, extractor_class: type, extractor_config: dict):
        return extractor_class(**extractor_config)
