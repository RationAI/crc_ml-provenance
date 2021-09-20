import importlib
import inspect
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple


def callable_has_signature(func: Callable, param_names: List[str]) -> bool:
    """
    Check any callable for the list of its parameter names.

    Parameters
    ----------
    func : Callable
        Any function or method to check.
    param_names : List[str]
        The list of parameter names checked against the signature of
        `func`.

    Return
    ------
    bool
        True if `param_names` contains exactly the parameter names of
        `func` (ignoring order), False otherwise.

    Raise
    -----
    TypeError
        When `func` is not Callable.
    """
    callable_params = list(inspect.signature(func).parameters)

    return (
            len(callable_params) == len(param_names)
            and all((name in callable_params) for name in param_names)
    )


def class_has_method(cls_object: type, method_name: str) -> bool:
    """
    Check whether a class object declares a method of given name.

    Illustrative example:

        class Dog:
            def __init__(self, age, color):
                self.age = age
                self.color = color

            def bark(self):
                return "Woof! " * self.age

        class_has_method(Dog, 'bark') => True
        class_has_method(Dog, 'meow') => False

    Parameters
    ----------
    cls_object : type
        The class object to check.
    method_name : str
        The name of the method to search `cls_object` for.

    Return
    ------
    bool
        True if `cls_object` has a method named `method_name`, False
        otherwise.
    """
    return (
        hasattr(cls_object, method_name)
        and callable(getattr(cls_object, method_name))
    )


def class_has_classmethod(cls_object: type, method_name: str) -> bool:
    """
    Check whether a class object declares a classmethod of given name.

    The same as `class_has_method`, with the additional constraint that the
    method is a class method.

    Parameters
    ----------
    cls_object : type
        The class object to check.
    method_name : str
        The name of the method to search `cls_object` for.

    Return
    ------
    bool
        True if `cls_object` has a classmethod named `method_name`, False
        otherwise.
    """
    return (
            class_has_method(cls_object, method_name)
            and type(cls_object.__dict__.get(method_name)) is classmethod
    )


def class_has_nonabstract_method(cls_object: type, method_name: str) -> bool:
    """
   Check whether a class object declares a non-abstract method of given name.

   The same as `class_has_method`, with the additional constraint that the
   method is not abstract method.

   Parameters
   ----------
   cls_object : type
       The class object to check.
   method_name : str
       The name of the method to search `cls_object` for.

   Return
   ------
   bool
       True if `cls_object` has a non-abstract method named `method_name`,
       False otherwise.
   """
    if not class_has_method(cls_object, method_name):
        return False

    method = getattr(cls_object, method_name)
    return getattr(method, '__isabstractmethod__', False) is False


def parse_module_and_class_string(descriptor: str) -> Tuple[str, str]:
    """
    Parse a class descriptor string into class and module descriptors.

    E.g.    'path.to.module.Class' -> ('path.to.module', 'Class')
            'Class' -> ('', 'Class')

    Ignores any dots surrounding the `descriptor`.

    Parameters
    ----------
    descriptor : str
        A class descriptor interpreted as the full path to a class.

    Return
    ------
    Tuple[str, str]
        A tuple containing the (module ID, class ID) strings.
    """
    dot_split = descriptor.strip('.').split('.')
    class_name = dot_split[-1]
    module_id = '.'.join(dot_split[:-1])
    return module_id, class_name


def load_class(class_descriptor: str) -> type:
    """
    Load a class by its class descriptor.

    Parameters
    ----------
    class_descriptor : str
        The full name of the class including its module namespace, e.g.
        'some.module.Class', where 'some.module' is the full module path and
        'Class' is the name of the class.

    Return
    ------
    Type
        The corresponding class.

    Raise
    -----
    AttributeError
        When the module on the corresponding module path exists, but does not
        define the expected class.
    ImportError
        When the module on the corresponding module path is nonexistent.
    """
    module_id, class_name = parse_module_and_class_string(class_descriptor)
    module = importlib.import_module(module_id)
    return getattr(module, class_name)


def run_method(obj: object, method_name: str, kwargs: dict) -> Any:
    """
    Runs a method of given object with passed parameters.

    Parameters
    ----------
    obj : object
        The object whose method should be run.
    method_name : str
        The name of the method to run.
    kwargs : dict
        The arguments to be passed to the method.

    Return
    ------
    Any
        The return value of the run method.

    Raise
    -----
    AttributeError
        When `obj` does not declare a (non-abstract) method `method_name`.
    """
    if not class_has_nonabstract_method(obj, method_name):
        raise AttributeError(
            f'object {obj} does not declare classmethod {method_name}'
        )
    return getattr(obj, method_name)(**kwargs)