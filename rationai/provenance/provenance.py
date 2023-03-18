# Standard Imports
import json
import hashlib
from typing import Dict
from typing import List
from pathlib import Path

# Third-party Imports
from pygit2 import Repository
import prov.model as prov
import pandas as pd

#Local Imports

# Namespace definitions
DEFAULT_NAMESPACE = 'master'
DEFAULT_NAMESPACE_URI = 'https://gitlab.ics.muni.cz/422328/pid-test/-/blob/master/'

PROVN_NAMESPACE = 'resolvable_prefix'
PROVN_NAMESPACE_URI = 'https://gitlab.ics.muni.cz/422328/pid-test/-/blob/master/provn/'

PID_NAMESPACE = 'pid'
PID_NAMESPACE_URI = 'https://gitlab.ics.muni.cz/422328/pid-test/-/blob/master/pid/'

ORGANISATION_DOI = '10.82521'
DOI_NAMESPACE = 'doi'
DOI_NAMESPACE_URI = f'https://doi.org/{ORGANISATION_DOI}/'

BUNDLE_PATHOL  = 'pathol.provn'
BUNDLE_PREPROC = 'preproc.provn'
BUNDLE_TRAIN   = 'train.provn'
BUNDLE_EVAL    = 'eval.provn'
BUNDLE_META    = 'meta.provn'

NAMESPACE_COMMON_MODEL = "cpm"
NAMESPACE_DCT = "dct"
NAMESPACE_PROV = "prov"



# Redirect all output to github repository folder
OUTPUT_DIR = Path('/home/jovyan/matejg/Data/prov_experiments/provn_test_pid')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)
if not (OUTPUT_DIR / 'provn').exists():
    (OUTPUT_DIR / 'provn').mkdir()


def get_bundle_identifier(filepath: Path):
    # Get Repository
    repo = Repository(filepath.parent)

    # Get remote URL
    remote_url = repo.remotes['origin'].url.replace(':', '/')
    remote_url = remote_url.split('@')[-1][:-4]

    # Get branch
    branch = repo.head.shorthand

    # Get relative path of file
    rel_path = filepath.relative_to(Path(repo.path).parent)

    git_url = str(Path(remote_url) / 'tree' / branch / rel_path)

    print(f'Remote URL: {remote_url}')
    print(f'Branch:     {branch}')
    print(f'Relpath:    {rel_path}')
    print('-------------------------')
    print(f'URI: {git_url}')

