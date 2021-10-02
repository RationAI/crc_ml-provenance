# Standard Imports
from __future__ import annotations
import json
import logging
from multiprocessing import Value
from pathlib import Path
from typing import Any, Optional
from typing import List

# Local Imports
from rationai.utils.config import ConfigProto

# Logger
log = logging.getLogger('create_map_config')


class CreateMapConfig(ConfigProto):
    """Iterable config for create map.

    The supplied config consists of two parts:

        - `_global` group is a mandatory key specifying default values. Parameters
        that are either same for every input or change only for some inputs
        should go in here.

        - One or more named groups. Each group custom group contains a list
        defining input files and parameters specific to these files. The
        value of these parameters will override the value of parameter
        defined in `_global` group.

    The config is an iterable object. At every iteration the CreateMapConfig
    first defaults back to `_global` values before being overriden by the
    input specific parameters.
    """
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

    def __iter__(self) -> CreateMapConfig:
        """Populates the config parameters with default values.

        Returns:
            CreateMapConfig: CreateMapConfig with `_global` values.
        """
        log.info('Populating default options.')
        with open(self.config_fp, 'r') as json_r:
            config = json.load(json_r)['slide-converter']

        # Set config to default state
        self.__set_options(config.pop('_global'))
        self._default_config = {}

        # Prepare iterator variable
        self._groups = config
        return self

    def __next__(self) -> CreateMapConfig:
        """First resets back default values before overriding the input specific
        parameters.

        Raises:
            StopIteration: No more input directories left to be processed

        Returns:
            CreateMapConfig: Fully populated CreateMapConfig ready to be processed.
        """
        if not (self._groups or self._cur_group_configs):
            raise StopIteration
        # For each input dir we only want to override
        # attributes explicitely configured in JSON file.
        self.__reset_to_default()
        if not self._cur_group_configs:
            self.__get_next_group()
        self.__set_options(self._cur_group_configs.pop())
        self.__validate_options()

        log.info(f'Now processing ({self.group}):{self.slide_dir}')
        return self

    def __set_options(self, partial_config: dict) -> None:
        """Iterates over the variable names and values pairs a setting
        the corresponding instance variables to these values.

        Args:
            config (dict): Partial configuration specifying variables
                           and values to be overriden
        """
        for k,v in partial_config.items():
            self.__set_option(k, v)

    def __set_option(self, k: str, v: Any) -> None:
        """Sets instance variable `k` with values `v`.

        Args:
            k (str): name of the instance variable
            v (Any): value to be set
        """
        if hasattr(self, k):
            self._default_config[k] = getattr(self, k)
            setattr(self, k, v)
        else:
            log.warning(f'Attribute {k} does not exist.')

    def __reset_to_default(self) -> None:
        """Reverts the overriden values back to the default values.
        """
        # Reset to global state
        while self._default_config:
            k,v = self._default_config.popitem()
            setattr(self, k, v)

    def __get_next_group(self) -> None:
        """Retrieves the next named group from the JSON config.
        """
        group, configs = self._groups.popitem()
        self.group = group
        self._cur_group_configs = configs

    def __validate_options(self) -> None:
        """Converts string paths to Path objects.
        """
        # Path attributes
        self.slide_dir = Path(self.slide_dir)
        if self.label_dir:
            self.label_dir = Path(self.label_dir)
        self.output_path = Path(self.output_path)

