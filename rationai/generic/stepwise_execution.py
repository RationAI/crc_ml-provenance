"""
Handling of executions of runs' steps.

Key concepts:
- StepConfig: a container for configuration information of individual steps,
              responsible for ensuring the correctness of config data and
              exposing it to the rest of the code
- StepInterface: an abstract class which should be implemented in order to
                 treat the resulting object as a pipeline (run) step
- StepExecutor: handles the general running of the pipeline - keeps ordering
                and runs steps (StepInterface objects) one-by-one

- step key: identifies a step (a string, e.g. "train", "test", etc.)
- contextual steps: steps which are identified by a step key in the format
                    "context.method"
                    in such a case, "context" describes the context (step
                    object), while "method" identifies the step (method of the
                    step object to be run).
- context key: is the first part before a period of a contextual step; for
               non-contextual steps, the context key is the same as the whole
               step key (in other words, we may recognize whether a key is
               contextual or not by seeing whether its context key equals the
               whole step key)
"""
from __future__ import annotations

import abc
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NoReturn, Optional

from rationai.utils import utils, DirStructure

log = logging.getLogger('step-exec')


def initialize_step(
        params: dict, step_config: StepConfig, dir_structure: DirStructure
) -> Optional[StepInterface]:
    """
    Initialize a StepInterface instance.

    RAI_UNTESTED

    Parameters
    ----------
    params : dict
        Overall parameters of the run.
    step_config : StepConfig
        Step initialization and execution-specific parameters.
    dir_structure : DirStructure
        The shared data directory structure for the run in which the step is to
        be executed.

    Return
    ------
    Optional[StepInterface]
        An initialized instance of the StepInterface, or None if a problem
        occurs.
    """
    log.info(f'Initializing {step_config.class_id}')
    cls = utils.load_class(step_config.class_id)

    if not issubclass(cls, StepInterface):
        log.error(
            f'Class {cls} must implement StepInterface to execute as a step.'
        )
        return None

    try:
        return cls.from_params(
            params=deepcopy(params),
            self_config=deepcopy(step_config.init_params),
            dir_structure=dir_structure
        )
    except Exception as ex:
        log.error(
            f'Failed to init {step_config.class_id}. Full stack trace: {ex}'
        )
        return None


def to_context_key(step_key: str) -> str:
    """
    Get corresponding context key from a step key.

    This function returns '<part1>' for input '<part1>.<part2>'. For input
    such as '<part1>', this same input is returned as output.

    Parameters
    ----------
    step_key : str
        The step key (identifier) from which to extract context key.

    Return
    ------
    str
        Either the context key (if `step_key` is contextual), or the `step_key`
        itself.

    Raise
    -----
    ValueError
        When `step_key` is malformed.
    """
    dot_split = step_key.split('.')
    if len(dot_split) > 2 or any((part == '') for part in dot_split):
        raise ValueError('malformed step key expression')

    return dot_split[0]


@dataclass
class StepConfig:
    """
    Container for configuration information for a single step of run.

    This may be parsed from the step definitions dictionary, which describes:
    - the step_key
    - initialization configuration: the step class and init parameters
    - method to execute and parameters to that method.

    The exact expected format of the step definitions dictionary is as follows:

    "step_definitions": {
        "step_key": {
            "init": {
                "class_id": str  # absolute import path of the class
                "config": dict   # __init__ attrs of the class (optional)
            },
            "exec": {
                "method": str   # method to be run
                "kwargs": dict  # kwargs of the method (optional)
            }
        },
        ...
    }

    Attributes
    ----------
    step_key : str
        The step identifier.
    context_key : str
        The identifier of the step context (the StepInterface implementation)
        for contextual objects. For one-off objects, this is the same as
        `step_key`.
    class_id : str
        The absolute import path of the step (context) class.
    init_params : str
        Parameters to the constructor of the step class.
    exec_method : str
        The name of the method to be run on the step (context) class.
    exec_kwargs : str
        The parameters to be passed to `exec_method` when it is run.
    """
    step_key: str
    context_key: str
    class_id: str
    init_params: dict[str, Any]
    exec_method: str
    exec_kwargs: dict[str, Any]

    @classmethod
    def from_step_definitions(cls, step_key: str, step_definitions: dict) -> Optional[StepConfig]:
        """
        Parse StepConfig from a 'step_definitions' dictionary.

        Parameters
        ----------
        step_key : str
            The key of the step whose configuration to parse.
        step_definitions : dict
            The dictionary of step definitions (configurations) to read from.
            See the class docstring for more information.

        Return
        ------
        Optional[StepConfig]
            The parsed StepConfig information or None when an error occurs
            during parsing.
        """
        try:
            step_definition = step_definitions[step_key]
        except KeyError:
            log.error(f'Step "{step_key}" not found in step_definitions.')
            return None

        try:
            context_key = to_context_key(step_key)
        except ValueError:
            log.error(f'The identifier of step "{step_key}" is malformed.')
            return None

        class_id, init_params = cls._parse_init_info(step_key, step_definition)
        exec_method, exec_params = cls._parse_exec_info(
            step_key, step_definition
        )

        if class_id is None or exec_method is None:
            return None

        return cls(
            step_key,
            context_key,
            class_id,
            init_params,
            exec_method,
            exec_params
        )

    @property
    def is_contextual_step(self) -> bool:
        """
        Check whether the step is contextual.

        Return
        ------
        bool
            True when the corresponding step is contextual, False otherwise.

        Raise
        -----
        ValueError
            When `self.step_key` is malformed.
        """
        return self.context_key != self.step_key

    @staticmethod
    def _parse_init_info(step_key: str, step_definition: dict):
        try:
            init_info = step_definition['init']
        except KeyError:
            log.error(
                f'No initialization info for step "{step_key}" provided.'
            )
            return None, None

        class_id = init_info.get('class_id')
        init_params = init_info.get('config', dict())

        if class_id is None:
            log.error(f'No class_id for step "{step_key}" provided.')
            return None, None

        return class_id, init_params

    @staticmethod
    def _parse_exec_info(step_key: str, step_definition: dict):
        try:
            exec_info = step_definition['exec']
        except KeyError:
            log.error(f'No execution info for step "{step_key}" provided.')
            return None, None

        exec_method = exec_info.get('method')
        exec_params = exec_info.get('kwargs', dict())

        if exec_method is None:
            log.error(f'No method to run for step "{step_key}" provided.')
            return None, None

        return exec_method, exec_params


class StepInterface(abc.ABC):
    """
    Interface for classes that are runnable as pipeline steps.

    Subclass requirements:
        - must implement:
            - @classmethod from_params(params: dict, self_config: dict)
            - method continue_from_run()
        - A step is initialized if issubclass(cls, StepInterface) is True
        - A step is run by specifying the method to execute.
          Optionally, the method's keyword arguments can be specified
          but have to be JSON serializable.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            utils.class_has_classmethod(subclass, 'from_params')
            and utils.callable_has_signature(
                subclass.from_params, ['params', 'self_config', 'dir_structure']
            )
            and utils.class_has_nonabstract_method(
                subclass, 'continue_from_run'
            )
            and utils.callable_has_signature(
                subclass.continue_from_run, ['self']
            )
        )

    @classmethod
    @abc.abstractmethod
    def from_params(cls, self_config: dict, params: dict, dir_structure: DirStructure) -> StepInterface:
        """
        Factory method used by `StepExecutor` to initialize a step.

        Subclasses have to implement this.

        Parameters
        ----------
        self_config : dict
            Configuration specific for this step.
        params : dict
            The general configuration information.
        dir_structure : DirStructure
            The shared data directory structure for the run in which this step
            is to be executed.
        """
        raise NotImplementedError(
            'Pipeline Step has to implement from_params classmethod'
        )

    @abc.abstractmethod
    def continue_from_run(self) -> NoReturn:
        """
        Prepare artifacts from a previous pipeline run such that they can be
        used in this step.
        """
        raise NotImplementedError(
            'Pipeline Step has to implement continue_from_run method'
        )


class StepExecutor:
    """
    Generic executor of pipeline steps.
    Iterates over an array of declared steps and attempts to run them.

    Requirements for classes runnable as "steps":
        1. Implement StepInterface
        2. Self initialize via from_params factory method.
        3. Contain a method that is runnable with JSON serializable kwargs.

    Config usage example:
        Array "ordered_steps" contains custom user-defined keys
        that represent the pipeline steps.
        Dictionary "step_definitions" contains a definition for each key.

        Additionally, multiple steps can be run using one instance.
        Define the behavior using a dot notation where the prefix
        delimited by a dot designates that the steps belong to the same object
        and resources should be released only after its last usage.

        Template:
         - "ordered_steps": ["step_key", "exp.train", "exp.test", "visualize"]
         - "step_definitions": see `StepConfig` documentation
    """
    context: dict[str, Optional[StepInterface]]
    current_step_idx: int
    params: dict[str, Any]
    step_definitions: dict
    step_keys: list[str]

    def __init__(self, step_keys: list[str], step_definitions: dict, params: dict, dir_structure: DirStructure):
        self.current_step_idx = -1  # starting index
        self.step_keys = step_keys[:]
        self.step_definitions = deepcopy(step_definitions)
        self.params = deepcopy(params)
        self.dir_structure = dir_structure

        # Stores step instances for re-usage
        self.context = dict()

    def next_step_key(self) -> Optional[str]:
        try:
            return self.step_keys[self.current_step_idx + 1]
        except IndexError:
            # current step is the last step defined
            return None

    def run_next(self) -> NoReturn:
        """Loads next class in the queue and runs its specified method.
        Returns a peek at the next item in the iterator
        or None when all the steps are performed.
        """
        step_key = self.next_step_key()
        if step_key is None:
            return False

        self.current_step_idx += 1

        step_config = StepConfig.from_step_definitions(
            step_key, self.step_definitions
        )

        if not step_config:
            return False

        step_instance = self._load_step_instance(step_config)
        if step_instance is None:
            return False

        try:
            utils.run_classmethod(
                step_instance.__class__,
                step_config.exec_method,
                step_config.exec_kwargs
            )
        except Exception as ex:
            log.error(f'Failed step run "{step_key}", full stack trace: {ex}')
            return False

        self._free_up_context_if_possible(step_config)
        return self.next_step_key() is not None

    def run_all(self) -> NoReturn:
        while self.run_next():
            pass

    def _load_step_instance(self, step_config: StepConfig) -> Optional[StepInterface]:
        """
        Initialize step instance.

        If contextual, the context gets cached for later use.

        RAI_UNTESTED

        Parameters
        ----------
        step_config : StepConfig
            Parsed configuration information for the step to be loaded.

        Return
        ------
        Optional[StepInterface]
            An instance of a StepInterface object, or None if the object fails
            to instantiate.
        """
        if step_config.is_contextual_step:
            return self._load_contextual_instance(step_config)

        return initialize_step(self.params, step_config, self.dir_structure)

    def _load_contextual_instance(self, step_config: StepConfig) -> Optional[StepInterface]:
        """
        Initialize contextual step instance and save it to context cache.

        RAI_UNTESTED

        Parameters
        ----------
        step_config : StepConfig
            Parsed configuration information for the step to be loaded.

        Return
        ------
        Optional[StepInterface]
            An instance of a StepInterface object, or None if the object fails
            to instantiate.
        """
        if step_config.context_key not in self.context:
            new_instance = initialize_step(self.params, step_config, self.dir_structure)
            log.debug(f'Saving {new_instance.__class__} to context.')
            self.context[step_config.context_key] = new_instance

        return self.context[step_config.context_key]

    def _free_up_context_if_possible(self, step_config: StepConfig) -> NoReturn:
        """
        Delete a step context if it will not appear again in the run.

        Non-contextual steps are ignored.

        RAI_UNTESTED

        Parameters
        ----------
        step_config : StepConfig
            Parsed configuration information for the step whose context is to
            be freed.
        """
        if not step_config.is_contextual_step:
            return

        if self._is_last_occurrence(step_config.context_key):
            log.debug(f'Releasing {step_config.context_key}')
            del self.context[step_config.context_key]

    def _is_last_occurrence(self, context_key: str) -> bool:
        """
        Check if a context key will appear anywhere in the remaining steps.

        Parameters
        ----------
        context_key : str
            The context key to look for in the remaining steps to be run.

        Return
        ------
            False if there are any remaining steps with given `context_key`,
            True otherwise.
        """
        return context_key not in map(
            to_context_key, self.step_keys[self.current_step_idx + 1:]
        )
