import abc
from pathlib import Path

# Type hints
from typing import Dict
from typing import Generator
from typing import NoReturn
from typing import TypeVar

T = TypeVar('T')

class ConverterBase(metaclass=abc.ABCMeta):
    """Formal interface for data conversion"""

    def __init__(self,
                 source_dir: Path,
                 output_base_dir: Path,
                 max_workers: int = 1):
        self.source_dir      = source_dir
        self.output_base_dir = output_base_dir

        # (Default) value 1 disables multiprocessing
        self.max_workers     = max_workers

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_batch_generator') and
                callable(subclass.get_batch_generator) and
                hasattr(subclass, 'convert') and
                callable(subclass.convert) and
                hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    @abc.abstractmethod
    def get_batch_generator(self) -> Generator[T, None, None]:
        """Returns generator which yields input for convert method."""
        raise NotImplementedError

    @abc.abstractmethod
    def convert(self, input_dict: T) -> NoReturn:
        """Converts input to desired format.

        If multiple workers are available,
        this method is mapped to self.get_batch_generator()"""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self) -> NoReturn:
        """Runs the conversion.
        Distributes computation if multiple workers are available """
        raise NotImplementedError

