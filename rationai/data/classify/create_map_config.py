# Standard Imports
import json
import logging
from multiprocessing import Value
from pathlib import Path
from typing import Optional
from typing import List

# Local Imports
from rationai.utils.config import ConfigProto

log = logging.getLogger('create_map_config')

class CreateMapConfig(ConfigProto):
    def __init__(self, config_fp):
        self.config_fp = config_fp

        # Input Path Parameters
        self.slide_dir = None
        self.label_dir = None
        self.pattern = None

        # Output Path Parameters
        self.output_path = None
        self.group = None

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
        self.negative_mode = False
        self.strict_mode = False
        self.force = False

        # Paralellization Parameters
        self.max_workers = None

        # Holding changed values
        self._default_config = {}

        # Iterable State
        self._groups = None
        self._cur_group_configs = []

    def __iter__(self):
        with open(self.config_fp, 'r') as json_r:
            config = json.load(json_r)['slide-converter']

        # Set config to default state
        self.__set_options(config.pop('_global').items())

        # Prepare iterator variable
        self._groups = config
        return self

    def __next__(self):
        if not (self._groups or self._cur_group_configs):
            raise StopIteration

        # For each input dir we only want to override
        # attributes explicitely configured in JSON file.
        self.__reset_to_default()
        if not self._cur_group_configs:
            self.__get_next_group()
        self.__set_options(self._cur_group_configs.pop())

        return self

    def __set_options(self, config):
        for k,v in config.items():
            self.__set_option(k, v)

    def __set_option(self, k, v):
        if hasattr(self, k):
            self._default_config[k] = getattr(self, k)
            setattr(self, k, v)
        else:
            log.warning(f'Attribute {k} does not exist.')

    def __reset_to_default(self):
        # Reset to global state
        while self._default_config:
            k,v = self._default_config.popitem()
            setattr(self, k, v)

    def __get_next_group(self):
        group, configs = self._groups.popitem()
        self.group = group
        self._cur_group_configs = configs



