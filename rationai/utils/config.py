class ConfigProto():
    def __init__(self, json_filepath, json_dict=None):
        self.json_filepath = json_filepath
        if json_dict is None:
            self.json_dict = self.__load_fron_file(json_filepath)
        self.__parse()

    def __load_fron_file(self, json_filepath):
        with open(json_filepath, 'r') as json_finput:
            return json.load(json_finput)

    def __parse(self):
        raise NotImplemented("ConfigProto.parse() not implemented.")