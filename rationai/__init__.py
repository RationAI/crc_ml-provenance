import logging

# Defines global logger format
FORMAT = '[%(asctime)s][%(levelname)-5.5s][%(name)-.20s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S')

__all__ = [
    'data', 'datagens', 'eval', 'generic', 'training', 'utils', 'visual'
]
