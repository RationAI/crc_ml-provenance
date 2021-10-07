import importlib

def get_class(full_path):
    """Retrieves a class given fully qualified path for a class."""
    module_name, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)