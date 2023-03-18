from pathlib import Path
import json
import rocrate

from rocrate.rocrate import ROCrate

from rationai.provenance.rocrate.ro_modules import WSI_Collection
from rationai.provenance.rocrate.ro_modules import HistopatEntity
from rationai.provenance.rocrate.ro_modules import HistopatScript
from rationai.provenance.rocrate.ro_modules import CPMProvenanceFile


def rocrate_module(crate, log_fp, meta_log_fp, prov_dict, meta_prov_dict, config_dict):
    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#train_script:{prov_dict["eid"]}', properties={
        'endTime': prov_dict['train']['end'],
        'name': 'VGG16 Training'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    instrument = crate.add(HistopatScript(crate, prov_dict['git_commit_hash'], prov_dict['script'], properties={'name': 'VGG16 Python Training Script'}))
    ce_convert.instrument = instrument

    # Create and Map Input Configuration File Entity
    ce_convert['object'] += [crate.add_file(prov_dict['config_file'], properties={
        'name': 'Input configuration file',
        'encodingFormat': 'text/json'
    })]

    # Create and Map Input Dataset File Entity
    ce_convert['object'] += [crate.add_file(config_dict['configurations']['datagen']['data_sources']['_data'], properties={
        'name': 'Dataset of ROI Indices',
        'encodingFormat': 'text/json'
    })]
    
    
    # Create and Map Output File Entities
    # Output provenance log
    prov_log = crate.add_file(str(log_fp), properties={
        'name': 'Experiment Run Log',
        'encodingFormat': 'application/json'
    })
    ce_convert['result'] += [prov_log]

    for iter_id, iter_data in prov_dict['iters'].items():
        if 'checkpoints' in iter_data:
            for save_data in iter_data['checkpoints'].values():
                if save_data['valid']:
                    ckpt_file = Path(save_data['filepath'])
                    for ckpt_file_part in ckpt_file.parent.glob(f'{ckpt_file.name}*'):
                        assert ckpt_file_part.exists(), 'Checkpoint part does not exist does not exist.'
                        ce_convert['result'] += [crate.add_file(ckpt_file_part, dest_path=f'model/weights/{ckpt_file_part.name}', properties={
                            'name': 'saved training weights'
                        })]
                    
    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#train_script:{meta_prov_dict["eid"]}:CPM-provgen', properties={
        'name': 'CPM Compliant Training Provenanace Generation Execution',
        'description': 'CPM compliant provenance generation for training.'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    assert Path(meta_prov_dict['script']).exists(), 'train provgen script does not exist'
    instrument = crate.add(
        HistopatScript(
            crate,
            meta_prov_dict['git_commit_hash'],
            meta_prov_dict['script'],
            properties={
                'name': 'Training Provenanace Generation Python Script',
                '@type': ['File', 'SoftwareSourceCode'],
                'description': 'A python script that translates the computation log files into CPM compliant provenance file.'
            }
        )
    )
    ce_convert.instrument = instrument

    ce_convert['object'] += [prov_log]

    # Create and Map Output Entities
    provn_entity = crate.add(
        CPMProvenanceFile(
            crate,
            Path(meta_prov_dict['output']['remote_provn']),
            properties={
                'name': 'Training Provenanace CPM File',
                '@type': ['File', 'CPMProvenanceFile'],
                'description': 'CPM compliant provenance file generated based on the computation log file.',
                'encodingFormat': ['text/provenance-notation', {'@id': 'http://www.w3.org/TR/2013/REC-prov-n-20130430/'}],
                'about': []
            }
        )
    )
    
    assert Path(meta_prov_dict['output']['local_png']).exists(), 'train PNG provn does not exist'
    provn_png_entity = crate.add(
        CPMProvenanceFile(
            crate,
            Path(meta_prov_dict['output']['remote_png']),
            properties={
                'name': 'PNG visualization of Provenanace CPM File',
                '@type': ['File'],
                'description': 'PNG visualization of a CPM compliant provenance file generated based on the computation log file.',
                'encodingFormat': 'image/png',
                'about': []
            }
        )
    )
    
    assert meta_log_fp.exists(), 'train meta_log_fp does not exist'
    provn_log_entity = crate.add_file(meta_log_fp, properties={
        '@type': ['File'],
        'name': 'Train provgen log file',
        'encodingFormat': 'text/json',
        'description': 'Log file for provenance generation.',
        'about': []
    })
    ce_convert['result'] += [provn_entity, provn_png_entity, provn_log_entity]
    provn_entity['about'] += [ce_convert]
    provn_log_entity['about'] += [ce_convert]
    provn_png_entity['about'] += [provn_entity]
    
    return crate


def rocrate_train(crate, meta_log_fp):    
    with meta_log_fp.open('r') as json_in:
        meta_prov_dict = json.load(json_in)
        
    log_fp = Path(meta_prov_dict['input']['log'])
    with log_fp.open('r') as json_in:
        prov_dict = json.load(json_in)
    
    with Path(prov_dict['config_file']).resolve().open('r') as config_in:
        config_dict = json.load(config_in)
        
    crate = rocrate_module(crate, log_fp, meta_log_fp, prov_dict, meta_prov_dict, config_dict)
    return crate