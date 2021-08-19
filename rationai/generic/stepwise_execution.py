from __future__ import annotations

import abc
import logging
from copy import deepcopy
from typing import (
    List,
    NoReturn,
    Optional
)

from rationai.utils import utils

log = logging.getLogger('step-exec')


def extract_context_key(step_key: str) -> str:
    """
    Get context key from step key.

    A step key identifies an execution step. Potentially, a step key can have
    the format '<part1>.<part2>'. In this case, <part1> describes the context
    (step object), while <part2> identifies the step (method of the step
    object to be run).

    This function returns '<part1>' for input '<part1>.<part2>'. For input
    '<part1>', this same input is returned as output.

    Parameters
    ----------
    step_key : str
        The step key (identifier).

    Return
    ------
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


def is_step_contextual(step_key: str) -> bool:
    """
    Check whether a step key is contextual.

    Parameters
    ----------
    step_key : str
        A step key (identifier).

    Return
    ------
    bool
        True when `step_key` is contextual, False otherwise.

    Raise
    -----
    ValueError
        When `step_key` is malformed.
    """
    return extract_context_key(step_key) != step_key


def initialize_step(params: dict, step_config: dict) -> Optional[StepInterface]:
    """
    Initialize a StepInterface instance.

    RAI_UNTESTED

    Parameters
    ----------
    params : dict
        Overall parameters of the run.
    step_config : dict
        Step initialization and execution-specific parameters.

    Return
    ------
    Optional[StepInterface]
        An initialized instance of the StepInterface, or None if a problem
        occurs.
    """
    try:
        class_descriptor: str = step_config['init']['class_id']
    except KeyError:
        log.error('Could not obtain class descriptor from step_config.')
        return None

    log.info(f'Initializing {class_descriptor}')
    cls = utils.load_class(class_descriptor)

    if not issubclass(cls, StepInterface):
        log.error(
            f'Class {cls} must implement StepInterface to execute as a step.'
        )
        return None

    try:
        return cls.from_params(
            params=deepcopy(params),
            self_config=deepcopy(step_config['init'].get('config', dict()))
        )
    except Exception as ex:
        log.error(f'Failed to init {class_descriptor}. Full stack trace: {ex}')
        return None


class StepInterface(abc.ABC):
    """
    Interface for classes that are runnable as pipeline steps.

    Subclass requirements:
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
                subclass.from_params, ['params', 'self_config']
            )
            and utils.class_has_method(subclass, 'continue_from_run')
            and utils.callable_has_signature(subclass.continue_from_run, [])
        )

    @classmethod
    @abc.abstractmethod
    def from_params(cls, self_config: dict, params: dict) -> StepInterface:
        """
        Factory method used by `StepExecutor` to initialize a step.

        Subclasses have to implement this.

        Parameters
        ----------
        self_config : dict
            Configuration specific for this step.
        params : dict
            The general configuration information.
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
        that represet the pipeline steps.
        Dictionary "step_definitions" contains a definition for each key.

        Additionally, multiple steps can be run using one instance.
        Define the behavior using a dot notation where the prefix
        delimited by a dot designates that the steps belong to the same object
        and resources should be released only after its last usage.

        Template:
         - "ordered_steps": ["step_key", "exp.train", "exp.test", "visualize"]
         - "step_definitions": {
                "step_key": {
                    "init": {
                        "class_id": str  # absolute import path of the class
                        "config": dict   # __init__ attrs of the class (optional)
                    },
                    "exec": {
                        "method": str   # method to be run
                        "kwargs": dict  # kwargs of the method (optional)
                    }
                }
            }
    """

    def __init__(self, ordered_steps: List[str], step_definitions: dict, params: dict):
        self.current_step_idx = -1  # starting index
        self.step_keys = deepcopy(ordered_steps)
        self.step_definitions = deepcopy(step_definitions)
        self.params = deepcopy(params)

        # Stores step instances for re-usage
        self.context = dict()

    def next_step_key(self) -> Optional[str]:
        try:
            return self.step_keys[self.current_step_idx + 1]
        except IndexError:
            # current step is the last step defined
            return None

    def run_has_more_steps(self) -> bool:
        return self.next_step_key() is not None

    def run_next(self) -> NoReturn:
        """Loads next class in the queue and runs its specified method.
        Returns a peek at the next item in the iterator
        or None when all the steps are performed.
        """
        if not self.run_has_more_steps():
            return

        step_name = self.next_step_key()
        self.current_step_idx += 1

        if step_name not in self.step_definitions:
            log.info(f'Step "{step_name}" not found in step_definitions. Skipping ...')
            return

        step_instance = self._load_step_instance(step_name)
        if step_instance is None:
            return

        try:
            exec_info = self.step_definitions[step_name]['exec']
        except KeyError:
            log.error(f'No execution info for step "{step_name}" provided.')
            return

        exec_method = exec_info.get('method')
        kwargs = exec_info.get('kwargs', dict())

        try:
            utils.run_classmethod(step_instance.__class__, exec_method, kwargs)
        except Exception as ex:
            log.error(f'Failed step run "{step_name}", full stack trace: {ex}')
            return

        self._free_up_context_if_possible(step_name)
        return

    def run_all(self) -> NoReturn:
        while self.run_has_more_steps():
            self.run_next()

    def _load_step_instance(self, step_key: str) -> Optional[StepInterface]:
        """
        Initialize step instance.

        If contextual, the context gets cached for later use.

        RAI_UNTESTED

        Parameters
        ----------
        step_key : str
            The identifier of the step.

        Return
        ------
        Optional[StepInterface]
            An instance of a StepInterface object, or None if the object fails
            to instantiate.
        """
        if is_step_contextual(step_key):
            return self._load_contextual_instance(step_key)

        return initialize_step(
            self.params, self.step_definitions.get(step_key)
        )

    def _load_contextual_instance(self, step_key: str) -> Optional[StepInterface]:
        """
        Initialize contextual step instance and save it to context cache.

        RAI_UNTESTED

        Parameters
        ----------
        step_key : str
            The identifier of the step.

        Return
        ------
        Optional[StepInterface]
            An instance of a StepInterface object, or None if the object fails
            to instantiate.
        """
        context_key = extract_context_key(step_key)
        if context_key not in self.context:
            new_instance = initialize_step(
                self.params, self.step_definitions.get(step_key)
            )
            log.debug(f'Saving {new_instance.__class__} to context.')
            self.context[context_key] = new_instance

        return self.context[context_key]

    def _free_up_context_if_possible(self, step_key: str) -> NoReturn:
        """
        Delete a step context if it will not appear again in the run.

        Non-contextual steps are ignored.

        RAI_UNTESTED

        Parameters
        ----------
        step_key : str
            The step identifier.
        """
        if not is_step_contextual(step_key):
            return

        context_key = extract_context_key(step_key)

        if self._is_last_occurrence(context_key):
            log.debug(f'Releasing {context_key}')
            del self.context[context_key]

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
            extract_context_key, self.step_keys[self.current_step_idx + 1:]
        )
