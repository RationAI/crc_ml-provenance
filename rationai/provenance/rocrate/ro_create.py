import argparse
from pathlib import Path
import json

from rocrate.rocrate import ROCrate

from rationai.provenance.rocrate.ro_preproc import rocrate_preproc
from rationai.provenance.rocrate.ro_train import rocrate_train
from rationai.provenance.rocrate.ro_test import rocrate_test
from rationai.provenance.rocrate.ro_meta import rocrate_meta


def create_rocrate(input_log):
    
    # Load meta prov log
    with input_log.open('r') as json_in:
        meta_prov_log = json.load(json_in)
    
    crate = ROCrate()
    crate.root_dataset['mentions'] = []
    crate.metadata.extra_terms.update({
        'CPMMetaProvenanceFile': 'https://w3id.org/ro/terms/cpm#CPMMetaProvenanceFile',
        'CPMProvenanceFile': 'https://w3id.org/ro/terms/cpm#CPMProvenanceFile'
    })
    
    # Test provenance
    meta_test_log_fp = Path(meta_prov_log['input']['test'])
    test_log_fp = meta_test_log_fp.with_suffix('').with_suffix('.log')
    crate = rocrate_test(crate, test_log_fp, meta_test_log_fp)
        
    # Train provenance
    meta_train_log_fp = Path(meta_prov_log['input']['train'])
    train_log_fp = meta_train_log_fp.with_suffix('').with_suffix('.log')
    crate = rocrate_train(crate, train_log_fp, meta_train_log_fp)
        
    # Preprocess provenance
    meta_preproc_log_fp = Path(meta_prov_log['input']['preprocess'])
    preproc_log_fp = meta_preproc_log_fp.with_suffix('').with_suffix('.log')
    crate = rocrate_preproc(crate, preproc_log_fp, meta_preproc_log_fp)
    
    # Meta provenance
    crate = rocrate_meta(crate, input_log)
    
    # Save to disk
    crate.write(input_log.parent / 'rocrate')
    crate.write_zip(input_log.parent / 'rocrate.zip')
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--rocrate_log', type=Path, required=True, help='Meta prov log of an experiment run.')
    args = parser.parse_args()
    
    create_rocrate(args.rocrate_log)

    