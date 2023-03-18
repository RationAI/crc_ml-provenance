import argparse
from pathlib import Path
import json

from rocrate.rocrate import ROCrate
from rocrate.model.contextentity import ContextEntity

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
    
    # Conformance statements
    ro_cpm = ContextEntity(crate, 'https://w3id.org/cpm/ro-crate/0.1', properties={
        '@type': 'CreativeWork',
        'name': 'Common Provenance Model RO-Crate profile',
        'version': '0.1'
    })

    ro_wfrun = ContextEntity(crate, 'https://w3id.org/ro/wfrun/process/0.1', properties={
        '@type': 'CreativeWork',
        'name': 'Process Run Crate',
        'version': '0.1'
    })
    crate.add(ro_cpm, ro_wfrun)
    crate.root_dataset['conformsTo'] = [ro_cpm, ro_wfrun]
    
    # Test provenance
    meta_test_log_fp = Path(meta_prov_log['input']['eval'])    # provlog file
    crate = rocrate_test(crate, meta_test_log_fp)
        
    # Train provenance
    meta_train_log_fp = Path(meta_prov_log['input']['train'])
    crate = rocrate_train(crate, meta_train_log_fp)
        
    # Preprocess provenance
    meta_preproc_log_fp = Path(meta_prov_log['input']['preprocess'])
    crate = rocrate_preproc(crate, meta_preproc_log_fp)
    
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

    