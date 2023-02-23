from pathlib import Path
import json
import rocrate

from rocrate.rocrate import ROCrate

from rationai.provenance.rocrate.ro_modules import WSI_Collection
from rationai.provenance.rocrate.ro_modules import HistopatEntity
from rationai.provenance.rocrate.ro_modules import HistopatScript


def rocrate_module(crate, log_fp, meta_log_fp, prov_dict, meta_prov_dict, config_dict):
    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#test_script:{prov_dict["eid"]}', properties={
        'endTime': prov_dict['predict']['end'],
        'name': 'VGG16 Predict'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    instrument = crate.add(HistopatScript(crate, prov_dict['git_commit_hash'], prov_dict['script'], properties={'name': 'VGG16 Python Test Script'}))
    ce_convert.instrument = instrument

    # Create and Map Input Configuration File Entity
    ce_convert['object'] += [crate.add_file(prov_dict['config_file'], properties={
        'name': 'input configuration file',
        'encodingFormat': 'text/json'
    })]

    # Create and Map Input Dataset File Entity
    ce_convert['object'] += [crate.add_file(config_dict['configurations']['datagen']['data_sources']['_data'], properties={
        'name': 'input dataset',
        'encodingFormat': 'text/json'
    })]

    # Create and Map Output File Entities
    prov_log = crate.add_file(str(log_fp), properties={
        'name': 'output provenance log',
        'encodingFormat': 'application/json'
    })
    
    ce_convert['result'] += [
        # Output dataset
        crate.add_file(prov_dict['predictions']['prediction_file'], properties={
            'name': 'output dataset with predictions and metrics',
            'encodingFormat': 'application/x-hdf5'
        }),
        # Output provenance log
        prov_log
    ]
 
    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#test_script:{meta_prov_dict["eid"]}:CPM-provgen', properties={
        'name': 'CPM Compliant Test Provenanace Generation Execution',
        'description': 'CPM compliant provenance generation for testing.'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    assert Path(meta_prov_dict['script']).exists(), 'test provgen script does not exist'
    instrument = crate.add_file(meta_prov_dict['script'], properties={
        'name': 'Test Provenanace Generation Python Script',
        '@type': ['File', 'SoftwareSourceCode'],
        'description': 'A python script that translates the computation log files into CPM compliant provenance file.',
    })
    ce_convert.instrument = instrument

    ce_convert['object'] += [prov_log]

    # Create and Map Output Entities
    assert Path(meta_prov_dict['output']['provn']).exists(), 'test provn does not exist'
    provn_entity = crate.add_file(meta_prov_dict['output']['provn'], properties={
        '@type': ['File', 'CPMProvenanceFile'],
        'name': 'Test Provenanace CPM File',
        'encodingFormat': ['text/provenance-notation', {'@id': 'http://www.w3.org/TR/2013/REC-prov-n-20130430/'}],
        'description': 'CPM compliant provenance file generated based on the computation log file.',
        'about': []
    })
    
    assert Path(meta_prov_dict['output']['png']).exists(), 'test PNG provn does not exist'
    provn_png_entity = crate.add_file(meta_prov_dict['output']['png'], properties={
        '@type': ['File'],
        'name': 'PNG visualization of Provenanace CPM File',
        'encodingFormat': 'image/png',
        'description': 'PNG visualization of a CPM compliant provenance file generated based on the computation log file.',
        'about': []
    })
    
    assert meta_log_fp.exists(), 'test meta_log_fp does not exist'
    provn_log_entity = crate.add_file(meta_log_fp, properties={
        '@type': ['File'],
        'name': 'Test provgen log file',
        'encodingFormat': 'text/json',
        'description': 'Log file for provenance generation.',
        'about': []
    })
    ce_convert['result'] += [provn_entity, provn_log_entity, provn_png_entity]
    provn_entity['about'] += [ce_convert]
    provn_log_entity['about'] += [ce_convert]
    provn_png_entity['about'] += [provn_entity]

    return crate


def rocrate_test(crate, log_fp, meta_log_fp):
    with log_fp.open('r') as json_in:
        prov_dict = json.load(json_in)
        
    with meta_log_fp.open('r') as json_in:
        meta_prov_dict = json.load(json_in)
    
    with Path(prov_dict['config_file']).resolve().open('r') as config_in:
        config_dict = json.load(config_in)
        
    crate = rocrate_module(crate, log_fp, meta_log_fp, prov_dict, meta_prov_dict, config_dict)
    return crate