# Standard Imports
import argparse
from pathlib import Path
import json
import uuid
import os


# Third-party Imports
import prov.model as prov
import pygit2


# Local Imports
from rationai.provenance import PID_NAMESPACE_URI
from rationai.provenance import PROVN_NAMESPACE
from rationai.provenance import PROVN_NAMESPACE_URI
from rationai.provenance import DEFAULT_NAMESPACE_URI
from rationai.provenance import PID_NAMESPACE
from rationai.provenance import BUNDLE_PREPROC
from rationai.provenance import BUNDLE_TRAIN
from rationai.provenance import BUNDLE_EVAL
from rationai.provenance import BUNDLE_META
from rationai.provenance import OUTPUT_DIR


from rationai.utils.provenance import export_to_image
from rationai.utils.provenance import export_to_file


def prepare_document():
    document = prov.ProvDocument()
    
    # Declaring namespaces
    document.add_namespace(PROVN_NAMESPACE, PROVN_NAMESPACE_URI)
    document.set_default_namespace(DEFAULT_NAMESPACE_URI)
    return document


def get_preproc_provlog(provlog):
    # Get config filepath from provenance log of provenance
    with provlog.open('r') as json_in:
        cfg = json.load(json_in)
    
    # Find out the source of inputs
    cfg_fp = Path(cfg['input']['config'])
    with cfg_fp.open('r') as json_in:
        cfg = json.load(json_in)
    return Path(cfg['configurations']['datagen']['data_sources']['_data']).parent / f'{BUNDLE_PREPROC}.log'

def export_provenance(experiment_dir: Path) -> None:
    
    provn_filepath = OUTPUT_DIR / 'provn' / BUNDLE_META
    png_filepath = (OUTPUT_DIR / 'provn' / BUNDLE_META).with_suffix('.png')
    json_filepath = (OUTPUT_DIR / 'json' / BUNDLE_META).with_suffix('.json')
    
    # Provenance of provenance
    output_log = {
        'git_commit_hash': str(pygit2.Repository('.').revparse_single('HEAD').hex),
        'script': str(__file__),
        'eid': str(uuid.uuid4()),
        'input': {},
        'output': {
            'png': str(png_filepath),
            'local_provn': str(provn_filepath),
            'remote_provn': str(PID_NAMESPACE_URI + BUNDLE_META)
        }
    }
        
    doc = prepare_document()
    bndl = doc.bundle(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    
    module_ns_mapping = {
        'train': experiment_dir / f'{BUNDLE_TRAIN}.log',
        'eval': experiment_dir / f'{BUNDLE_EVAL}.log',
        'preprocess': get_preproc_provlog(experiment_dir / f'{BUNDLE_EVAL}.log')
    }
    
    for module, provlog in module_ns_mapping.items():
        output_log['input'][module] = str(provlog.resolve())
    
        b = bndl.entity(f'bundle_{module}', other_attributes={
            'prov:type': 'prov:bundle'
        })

        b_gen = bndl.entity(f'bundle_{module}_gen', other_attributes={
            'prov:type': 'prov:bundle'
        })

        bndl.specialization(b, b_gen)
        
    
    
    export_to_image(bndl, png_filepath)
    export_to_file(doc, provn_filepath, format='provn')
    export_to_file(doc, json_filepath, format='json')
    
    with open(experiment_dir / f'{BUNDLE_META}.log', 'w') as json_out:
        json.dump(output_log, json_out, indent=3)
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Configuration file')
    parser.add_argument('--eid', type=str, required=True, help='Experiment UUID')
    args = parser.parse_args()
    
    with args.config_fp.open('r') as json_in:
        cfg = json.load(json_in)
        
    experiment_dir = Path(cfg['output_dir']) / args.eid
    
    export_provenance(experiment_dir)