# Standard Imports

# Third-party Imports
import json

# Local Imports


class ConfigProto:
    """ConfigProto consumes and parses JSON configuration file."""
    def __init__(self, json_dict: dict):
        self.config = json_dict

    @classmethod
    def load_from_file(self, json_filepath):
        with open(json_filepath, 'r') as json_finput:
            json_config = json.load(json_finput)
        return self(json_config)

    def parse(self, config_fp):
        raise NotImplemented("ConfigProto.parse() not implemented.")
