import os

# directory for profiling logs
if not os.path.exists('data/imreg/logs'):
    os.makedirs('data/imreg/logs')

from rationai.imreg.main import ImageRegistration

__all__ = [
    'ImageRegistration'
]
