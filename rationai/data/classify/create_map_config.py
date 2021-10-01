# Standard Imports
import json
from multiprocessing import Value
from pathlib import Path
from typing import Optional
from typing import List

# Local Imports
from rationai.utils.config import ConfigProto

class CreateMapConfig(ConfigProto):
    def __init__(self, config_fp):
        self.config_fp = config_fp

        # Input Path Parameters
        self.input_data = None
        self.pattern = None

        # Output Path Parameters
        self.output_path = None

        # Tile Parameters
        self.tile_size = None
        self.step_size = None
        self.center_size = None

        # Resolution Parameters
        self.sample_level = None
        self.bg_level = None

        # Filtering Parameters
        self.include_keywords = None
        self.exclude_keywords = None
        self.min_tissue = None
        self.max_tissue = None
        self.disk_size = None

        # Tiling Modes
        self.negative = False
        self.strict = False
        self.force = False

        # Paralellization Parameters
        self.max_workers = None

        self.__parse(config_fp)

    def __parse(self, config_fp):
        with open(config_fp, 'r') as cfg_finput:
            config = json.load(cfg_finput)

        # Data parameters
        data_config = config['data']
        self.input_data = data_config['input_data']
        self.tile_size = int(data_config['tile_size'])
        self.center_size = int(data_config['center_size'])
        self.sample_level = int(data_config['sample_level'])
        self.bg_level = int(data_config['bg_level'])

        # Converter specific parameters
        converter_config = config['converter']
        self.pattern = str(converter_config['pattern'])
        self.output_path = Path(converter_config['output_path'])
        self.step_size = int(converter_config['step_size'])
        self.include_keywords = list(converter_config['include_keywords'])
        self.exclude_keywords = list(converter_config['exclude_keywords'])
        self.min_tissue = float(converter_config['min_tissue'])
        self.max_tissue = float(converter_config['max_tissue'])
        self.disk_size = int(converter_config['disk_size'])
        self.negative = bool(converter_config['negative_mode'])
        self.strict = self.__validate_strict(bool(converter_config['strict_mode']))
        self.force = bool(converter_config['force'])
        self.max_workers = int(converter_config['max_workers'])

    def __validate_strict(self, strict_enabled: bool) -> bool:
        """Strict mode requires an annotation file to produce meaningful results.

        Args:
            strict_enabled (bool): Strict mode enabled flag

        Returns:
            bool: Strict mode enabled flag
        """
        if strict_enabled:
            assert self.label_dir is not None, "Strict mode requires a valid annotation file."
        return strict_enabled

    def to_json(self, output_path):
        json_dict = {}
        json_dict['data'] = {}
        json_dict['converter'] = {}

        data_json_dict = json_dict['data']
        data_json_dict['input_data'] = self.input_data
        data_json_dict['tile_size'] = self.tile_size
        data_json_dict['center_size'] = self.center_size
        data_json_dict['sample_level'] = self.sample_level
        data_json_dict['bg_level'] = self.bg_level

        converter_json_dict = json_dict['converter']
        converter_json_dict['pattern'] = self.pattern
        converter_json_dict['output_path'] = str(self.output_path)
        converter_json_dict['step_size'] = self.step_size
        converter_json_dict['include_keywords'] = self.include_keywords
        converter_json_dict['exclude_keywords'] = self.exclude_keywords
        converter_json_dict['min_tissue'] = self.min_tissue
        converter_json_dict['max_tissue'] = self.max_tissue
        converter_json_dict['disk_size'] = self.disk_size
        converter_json_dict['negative_mode'] = self.negative
        converter_json_dict['strict_mode'] = self.strict
        converter_json_dict['force'] = self.force
        converter_json_dict['max_workers'] = self.max_workers

        with open(output_path, 'w') as json_fout:
            json.dump(json_dict, json_fout)



