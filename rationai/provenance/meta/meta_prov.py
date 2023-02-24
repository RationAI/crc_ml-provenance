# Standard Imports
import argparse
from pathlib import Path
import uuid
import json
import os
from pathlib import Path
import pygit2


# Third-party Imports


# Local Imports
from rationai.provenance import NAMESPACE_PREPROC
from rationai.provenance import NAMESPACE_EVAL
from rationai.provenance import NAMESPACE_TRAINING
from rationai.provenance import NAMESPACE_METAPROV
from rationai.provenance import prepare_document

from rationai.utils.provenance import export_to_image, export_to_provn


def get_preproc_provlog(provlog):
    # Get config filepath from provenance log of provenance
    with provlog.open('r') as json_in:
        cfg = json.load(json_in)
    
    # Find out the source of inputs
    cfg_fp = Path(cfg['input'])
    with cfg_fp.open('r') as json_in:
        cfg = json.load(json_in)
    return Path(cfg['configurations']['datagen']['data_sources']['_data']).parent / 'prov_preprocess.provn.log'

def export_provenance(experiment_dir: Path) -> None:
    
    # Provenance of provenance
    output_log = {
        'git_commit_hash': str(pygit2.Repository('.').revparse_single('HEAD').hex),
        'script': str(__file__),
        'eid': str(uuid.uuid4()),
        'input': {},
        'output': {
            'png': str((experiment_dir / 'meta_provenance.png').resolve()),
            'provn': str((experiment_dir / 'meta_provenance.provn').resolve())
        }
    }
        
    doc = prepare_document()
    bndl = doc.bundle(f'{NAMESPACE_METAPROV}:meta-provenance')
    
    module_ns_mapping = {
        'train': (experiment_dir / 'prov_train.provn.log', NAMESPACE_TRAINING),
        'test': (experiment_dir / 'prov_test.provn.log', NAMESPACE_EVAL),
        'preprocess': (get_preproc_provlog(experiment_dir / 'prov_test.provn.log'), NAMESPACE_PREPROC)
    }
    
    for module, (provlog, ns) in module_ns_mapping.items():
        output_log['input'][module] = str(provlog.resolve())
    
        b = bndl.entity(f'{ns}:bundle_{module}', other_attributes={
            'prov:type': 'prov:bundle'
        })

        b_gen = bndl.entity(f'{ns}:bundle_{module}_gen', other_attributes={
            'prov:type': 'prov:bundle'
        })

        bndl.specialization(b, b_gen)
        
    
    
    export_to_image(bndl, experiment_dir / 'meta_provenance.png')
    export_to_provn(doc, experiment_dir / 'meta_provenance.provn')
    
    with open(experiment_dir / 'meta_provenance.provn.log', 'w') as json_out:
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