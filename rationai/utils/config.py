"""
TODO: Missing docstring.
"""
import json


class ConfigProto:
    """
    TODO: Missing docstring.
    """
    def __init__(self, json_filepath, json_dict=None):
        self.json_filepath = json_filepath
        if json_dict is None:
            self.json_dict = self.__load_from_file(json_filepath)

        # TODO: Should this be here? With what parameters?
        self.__parse(None)  # TODO: replace None with proper params

    @staticmethod
    def __load_from_file(json_filepath):
        with open(json_filepath, 'r') as json_finput:
            return json.load(json_finput)

    # TODO: Why is this method private?
    #       What should be the parameters?
    def __parse(self, config_fp):
        raise NotImplemented("ConfigProto.parse() not implemented.")
