import importlib

def get_class(module_name, class_name):
    """Retrieves a class given fully qualified path for a class."""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)