"""
Decorator for aspect-oriented (AOP) logging.
"""
from pathlib import Path
from typing import Union, NoReturn


class Loggable:
    """
    Decorator for AOP logging.

    Logs messages of the format "<key>: <value>". Messages are logged either
    to the filepath in Loggable.PATH_TO_LOG, if this is set, or to standard
    output otherwise.

    - <value> is the return value of the decorated function
    - <key> is provided as a parameter to the decorator
    """
    PATH_TO_LOG = None

    def __init__(self, prefix: str):
        self.key_prefix = prefix

    def __call__(self, func, *args, **kwargs):
        """
        RAI_UNTESTABLE - logging to file is not unit-testable
        """
        def wrapper_loggable(*args, **kwargs):
            function_result = func(*args, **kwargs)

            log_message = f'{self.key_prefix}: {function_result}'

            if Loggable.PATH_TO_LOG is None:
                print(log_message)
            else:
                with open(Loggable.PATH_TO_LOG, 'a') as log_file:
                    log_file.write(log_message)

            return function_result

        return wrapper_loggable

    @classmethod
    def set_log_path(cls, log_path: Union[str, Path]) -> NoReturn:
        """
        Set global path used to log results of functions decorated by Loggable.

        This should be called before any logging actually takes place,
        otherwise the logs are output to stdout.
        """
        Loggable.PATH_TO_LOG = log_path
