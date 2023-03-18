import json
from pathlib import Path

from rationai.provenance.rocrate.ro_modules import HistopatEntity
from rationai.provenance.rocrate.ro_modules import HistopatScript
from rationai.provenance.rocrate.ro_modules import CPMProvenanceFile

def rocrate_prov(crate, meta_log_fp, meta_prov_dict):
    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#meta_provn_script:{meta_prov_dict["eid"]}:CPM-provgen', properties={
        'name': 'CPM Compliant Meta Provenanace Generation Execution',
        'description': 'CPM compliant meta provenance generation for.'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    instrument = crate.add(
        HistopatScript(
            crate,
            meta_prov_dict['git_commit_hash'],
            meta_prov_dict['script'],
            properties={
                'name': 'Meta Provenanace Generation Python Script',
                '@type': ['File', 'SoftwareSourceCode'],
                'description': 'A python script that translates the computation log files into CPM compliant provenance file.'
            }
        )
    )
    ce_convert.instrument = instrument

    # Dereference and Map Input Files
    for input_type, input_fp in meta_prov_dict['input'].items():
        crate_ref = crate.dereference(str(Path(input_fp).name))
        assert crate_ref is not None, f'Dereference failed for: {input_fp}'
        ce_convert['object'] += [crate_ref]
    
    # Create and Map Output Entities
    provn_entity = crate.add(
        CPMProvenanceFile(
            crate,
            Path(meta_prov_dict['output']['remote_provn']),
            properties={
                'name': 'Meta Provenanace CPM File',
                '@type': ['File', 'CPMMetaProvenanceFile'],
                'description': 'CPM compliant provenance file generated based on the computation log file.',
                'encodingFormat': ['text/provenance-notation', {'@id': 'http://www.w3.org/TR/2013/REC-prov-n-20130430/'}],
                'about': []
            }
        )
    )

    assert Path(meta_prov_dict['output']['png']).exists(), 'meta png does not exist'
    provn_png_entity = crate.add_file(meta_prov_dict['output']['png'], properties={
        '@type': ['File'],
        'name': 'Meta provenance PNG graph',
        'encodingFormat': 'image/png',
        'description': 'PNG visualization of a CPM compliant provenance.',
        'about': []
    })
    assert meta_log_fp.exists(), 'meta meta_log_fp does not exist'
    provn_log_entity = crate.add_file(meta_log_fp, properties={
        '@type': ['File'],
        'name': 'Preprocess provgen log file',
        'encodingFormat': 'text/json',
        'description': 'Log file for provenance generation.',
        'about': []
    })
    ce_convert['result'] += [provn_entity, provn_png_entity, provn_log_entity]
    provn_entity['about'] += [ce_convert]
    provn_png_entity['about'] += [ce_convert]
    provn_log_entity['about'] += [ce_convert]

    return crate

def rocrate_meta(crate, meta_log_fp): 
    with meta_log_fp.open('r') as json_in:
        meta_prov_dict = json.load(json_in)
        
    crate = rocrate_prov(crate, meta_log_fp, meta_prov_dict)
    return crate