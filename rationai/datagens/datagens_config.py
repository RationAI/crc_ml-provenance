# Standard Imports
import json

# Third-party Imports

# Local Imports
from rationai.utils.config import ConfigProto


class DatagenConfig(ConfigProto):

    def __init__(self, config_fp):
        self.config_fp = config_fp

        self.data_sources_config = None
        self.generators_config = None

    def __parse(self, config_fp):
        with open(config_fp, 'r') as cfg_finput:
            config = json.load(cfg_finput)

        self.data_source_config = config['data_sources']
        self.generators_config = config['generators']


