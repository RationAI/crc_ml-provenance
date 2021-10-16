"""Configuration parsing utilities."""
# Standard Imports
from __future__ import annotations

# Third-party Imports
import json

# Local Imports
from pathlib import Path
from typing import NoReturn


class ConfigProto:
    """ConfigProto consumes and parses JSON configuration file.

    Base class for configuration parsing classes.
    """
    def __init__(self, json_dict: dict):
        self.config = json_dict

    @classmethod
    def load_from_file(cls, json_filepath: Path) -> ConfigProto:
        """Load the configuration from a JSON file.

        Parameters
        ----------
        json_filepath : pathlib.Path
            Path to JSON configuration file.
        """
        with open(json_filepath, 'r') as json_finput:
            json_config = json.load(json_finput)
        return cls(json_config)

    def parse(self) -> NoReturn:
        """Parses self.config into the actual configuration."""
        raise NotImplemented("ConfigProto.parse() not implemented.")
