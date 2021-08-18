from __future__ import annotations

import abc
import importlib
import inspect
import logging
from copy import deepcopy
from typing import (
    List,
    NoReturn,
    Optional
)

from rationai.utils import utils

log = logging.getLogger('step-exec')


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

    def __init__(self,
                 ordered_steps: List[str],
                 step_definitions: dict,
                 params: dict):

        self.current_step_idx = -1  # starting index
        self.step_keys = deepcopy(ordered_steps)
        self.step_definitions = deepcopy(step_definitions)
        self.params = deepcopy(params)

        # Stores step instances for re-usage
        self.context = dict()

    def run_all(self) -> NoReturn:
        """Runs all steps"""
        while self.run_next():
            pass

    def peek_next(self) -> Optional[str]:
        """Returns a name of the next step"""
        next_step = self._next()

        if next_step:
            self.current_step_idx -= 1
        return next_step

    def run_next(self) -> Optional[str]:
        """Loads next class in the queue and runs its specified method.
        Returns a peek at the next item in the iterator
        or None when all the steps are performed.
        """

        step_name = self._next()
        if not step_name:
            return None
        if step_name not in self.step_definitions:
            log.info(f'Step "{step_name}" not found in step_definitions. Skipping ...')
            return self.peek_next()

        context_key = step_name.split('.')[0]

        # flag to release a step from self.context once it is not needed anymore
        release_context = False

        if context_key != step_name:
            # INIT & SAVE to self.context
            if context_key not in self.context:
                instance = self._init_class(self.step_definitions.get(step_name))
                log.debug(f'Saving {instance.__class__} to context')
                self.context[context_key] = instance

            # LOAD from self.context
            elif context_key in self.context:
                instance = self.context[context_key]

                # RELEASE from self.context
                if self._is_last_occurence(context_key):
                    release_context = True
        else:
            # INIT
            instance = self._init_class(self.step_definitions.get(step_name))

        if instance is None:
            return self.peek_next()

        # get method name to execute
        exec_method = self.step_definitions[step_name]['exec'].get('method')
        if not hasattr(instance.__class__, exec_method):
            log.info(f'Class {instance.__class__} '
                     f'does not have the method "{exec_method}".')
            log.info(f'Skipping execution of step with key "{step_name}"')
            return self.peek_next()

        # RUN METHOD
        kwargs = self.step_definitions[step_name]['exec'].get('kwargs', dict())
        getattr(instance, exec_method)(**kwargs)

        # free up resources
        if release_context:
            log.debug(f'Releasing {context_key}')
            del self.context[context_key]

        return self.peek_next()

    def _next(self) -> Optional[str]:
        """Returns key of the next step or None"""
        if self.current_step_idx + 1 <= len(self.step_keys) - 1:
            self.current_step_idx += 1
            return self.step_keys[self.current_step_idx]
        return None

    def _is_last_occurence(self, context_key: str) -> bool:
        """Returns True if context_key is not present in the remaining steps"""
        if self.current_step_idx == len(self.step_keys) - 1:
            return True

        # context key not in remaining steps context keys
        return context_key not in list(
            map(lambda x: x.split('.')[0],
                deepcopy(self.step_keys[self.current_step_idx + 1:])))

    def _init_class(self, step_config: dict) -> object:
        """Returns initialized class defined by step_config"""
        if step_config is None:
            log.info('step_config is None')
            return None

        # Parse module & class from string
        split = step_config['init']['class_id'].split('.')
        class_name = split[-1]
        module_id = '.'.join(split[:-1])

        log.info(f'Initializing {class_name}')

        # Import the class
        module = importlib.import_module(module_id)
        cls = getattr(module, class_name)

        if not issubclass(cls, StepInterface):
            log.info(f'Class {cls} should implement StepInterface to work as step')
            return None

        try:
            return cls.from_params(
                params=deepcopy(self.params),
                self_config=deepcopy(step_config['init'].get('config', dict()))
            )
        except Exception as e:
            log.info(f'Failed to init {class_name}. Skipping its execution. {e}')
            return None
