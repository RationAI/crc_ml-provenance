# Standard Imports
import json
import hashlib
from typing import Dict
from typing import List
from pathlib import Path

# Third-party Imports
import prov.model as prov
import pandas as pd

#Local Imports

# Namespace definitions
NAMESPACE_EVAL = "ns_eval"
NAMESPACE_TRAINING = "ns_training"
NAMESPACE_PREPROC = "ns_preprocessing"
NAMESPACE_PATHOLOGY = "ns_pathology"

NAMESPACE_COMMON_MODEL = "cpm"
NAMESPACE_DCT = "dct"
NAMESPACE_PROV = "prov"


def prepare_document():
    document = prov.ProvDocument()
    
    # Declaring namespaces
    document.add_namespace(NAMESPACE_PREPROC, "preproc_uri")
    document.add_namespace(NAMESPACE_PATHOLOGY, "pathology_uri")
    document.add_namespace(NAMESPACE_COMMON_MODEL, "cpm_uri")
    document.add_namespace(NAMESPACE_DCT, "dct_uri")
    document.add_namespace(NAMESPACE_EVAL, "eval_uri")
    document.add_namespace(NAMESPACE_TRAINING, "test_uri")
    document.add_namespace(NAMESPACE_PROV, 'prov_uri')
    
    document.set_default_namespace('http://example.org/0/')

    return document
