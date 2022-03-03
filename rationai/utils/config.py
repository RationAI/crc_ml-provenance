"""Configuration parsing utilities."""
# Standard Imports
from __future__ import annotations
from typing import Any

# Third-party Imports
import json

# Local Imports
from pathlib import Path
from typing import NoReturn

from rationai.utils.class_handler import get_class


class ConfigProto:
    """ConfigProto consumes and parses JSON configuration file.

    Base class for configuration parsing classes.

    Attributes
    ----------
    config : dict
        The dictionary containing the configuration to be parsed.
    """
    def __init__(self, json_dict: dict):
        self.config = json_dict
        self.components = None
        self.cls = None

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

    def parse(self, cfg_store: dict = None) -> NoReturn:
        """
        Parses self.config into the actual configuration.

        Args:
            cfg_store (dict): Contains key-value pairs of named configurations.
                Can be used to provide environment-like variables and settings.
                If you need any global variable, paths or flags, specify them in the configuration file as a named configuration
                and it will be availbale in this dictionary.
        """
        raise NotImplemented("ConfigProto.parse() not implemented.")



    
def parse_configs_recursively(cfg: dict | str | list, cfg_store: dict | str = None, ref_trace=None, parsed_config_store: dict = None):
    """
    This method recursively parses the configuration dictionary and builds the modules from it,
    maintaining the structure.
    Cyclic dependencies are detected through a module trace and an exception is raised.

    Args:
        cfg (dict | str | list): The configuration dictionary containing all the configurations that are
            referenced via string in the source configuration dictionary.
            If reference by name is used and the source for named configurations is None, exception is raised.
        cfg_store (dict | str): Contains key-value pairs of named configurations.
            Key is the configuration name and value is the configuration dictionary.
            When supplied, the configrations in teh source dict can also reference values in this dictionay by their key.
            Defaults to None.
        ref_trace (dict): Traces the named references so that infinite recursion is avoided.
            A dict data structure is used to store the trace to allow instant membership evaluation. The order is maintained through the dictionary values.
            Defaults to empty dict().
        parsed_config_store (dict): Dictionary containing already built modules.

    Returns:
        dict: The dictionary containing all the configurations defined by the source config.
    """
    if ref_trace is None:
        ref_trace = dict()
    if parsed_config_store is None:
        parsed_config_store = dict()

    # check in the named configuration store is a key and if so, retrieve it from the source config dict
    if isinstance(cfg_store, str):
        # if the source config is also just a reference, raise an error
        if isinstance(cfg, str):
            raise InvalidConfigurationException(key=cfg, trace=ref_trace, 
                message="You cannot reference the key-value configuration store by string reference while also define the source configuration by string reference.\n\
                         You have to supply a full dictionary for one of those.")
        else:
            # replace the reference string with the fully fledged dictionary retrieved from the config
            cfg_store = cfg[cfg_store]

    # if the config is defined by reference, retrieve the referenced dictionary from the named config store
    if isinstance(cfg, str):
        # check if it is not already built and if so, skip building and return the built instance
        if cfg in parsed_config_store:
            return parsed_config_store[cfg]

        # check if the reference is not cyclic
        if cfg in ref_trace:
            raise InvalidConfigurationException(key=cfg, trace=ref_trace, 
                message=f"The entry '{cfg}' describes a component which is currently being resolved. Cyclic references are not possible!")
        
        # check if the reference points to an existing entry
        if cfg not in cfg_store:
            raise InvalidConfigurationException(key=cfg, trace=ref_trace, 
                message=f"The entry '{cfg}' does not refer to any existing key in the supplied dictionary with named configurations!")
        

        # add the reference to the trace for all further recursive calls
        ref_trace[cfg] = len(ref_trace)
        parsed_config = parse_configs_recursively(cfg_store[cfg], cfg_store, ref_trace, parsed_config_store)
        # after module is complete, remove its key from the trace, so it does not cause trouble for other references
        del ref_trace[cfg]
        parsed_config_store[cfg] = parsed_config
        return parsed_config

    # if the config is defined as a list of configurations, just parse each element recursively
    elif isinstance(cfg, list):
        modules = list()
        for config_element in cfg:
            modules.append(parse_configs_recursively(config_element, cfg_store, ref_trace, parsed_config_store))
        return modules
    
    # if the config is a dictionary, just parse it as a component.
    elif isinstance(cfg, dict):
        # retrieve the component class
        component_class = get_class(cfg['_class'])
        component_cfg = None
        # if the component has a Config member class defined, use it to parse the component configuration
        if hasattr(component_class, 'Config') and issubclass(component_class.Config, ConfigProto):
            component_cfg = component_class.Config(cfg)
            component_cfg.cls = component_class  # save the class reference into the config so that we can use it later directly
            component_cfg.parse(cfg_store)  # parse the config, extract variables from cfg store, anything really
            #if there are any subcomponents, parse their configs too
            if '_components' in cfg:
                component_cfg.components = dict()
                for _name, _config in cfg['_components'].items():
                    component_cfg.components[_name] = parse_configs_recursively(_config, cfg_store, ref_trace, parsed_config_store)
            return component_cfg

        # if it is not a component class, just return the dict
        else:
            # skip arguments starting with "_"
            return cfg
          

def build_from_config(cfg: ConfigProto | dict) -> Any:
    """This is just a shorthand. You need to pass a ConfigProto instanco to a module's __init__ to get a configured module,
    but the config already knows which class it is from so we can just extract it from it and instantitate a the module directly
    from the config

    Args:
        cfg (ConfigProto): Configuration instance with everything set up

    Returns:
        Any: An instance of a configured module/object
    """
    if isinstance(cfg, dict):
        cls = get_class(cfg['_class'])
        return cls(**{k:v for k, v in cfg.items() if k != '_class'})
    else:
        return cfg.cls(cfg)




class InvalidConfigurationException(Exception):
    """
    This exception is used when the configuration is just wrong somehow.
    """
    def __init__(self, key: str, trace: dict(), message):
        self.key = key
        self.trace = [k for k, v in sorted(trace.items(), key=lambda item_pair: item_pair[1])]
        super().__init__(message + "\nReference trace: " + "->".join([f"*[{_key}]*"  if _key == key else f"({_key})" for _key in self.trace]))

