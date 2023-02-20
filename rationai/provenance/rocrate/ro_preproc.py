from pathlib import Path
import json
import rocrate

from rocrate.rocrate import ROCrate

from rationai.provenance.rocrate.ro_modules import WSI_Collection
from rationai.provenance.rocrate.ro_modules import HistopatEntity
from rationai.provenance.rocrate.ro_modules import HistopatScript

def process_group(crate, group, json_group, pattern, debug=False):
    datasets = []
    for sub_group in json_group:
        slide_dir = Path(sub_group['slide_dir'])
        label_dir = None if sub_group['label_dir'] is None else Path(sub_group['label_dir'])
    
        wsi_list = list(slide_dir.glob(pattern))

        # Creating collection entities for each WSI
        for wsi in wsi_list:
            wsi_collection = set(slide_dir.glob(f'{wsi.stem}*'))
            if label_dir is not None:
                wsi_collection = wsi_collection.union(set(label_dir.glob(f'{wsi.stem}*')))
            
            d = crate.add(WSI_Collection(crate, source=f'#{wsi.stem}'))
            for fp in wsi_collection:
                if fp.is_file():
                    e = crate.add_file(str(fp), dest_path=f'wsi/{group}/{fp.name}')
                else:
                    e = crate.add_dataset(str(fp), dest_path=f'wsi/{group}/{fp.name}')
                d['hasPart'] += [e]
                if fp == wsi:
                    d['mainEntity'] = e
            datasets.append(d)
        
    return datasets

def rocrate_module(crate, log_fp, meta_log_fp, prov_dict, meta_prov_dict, config_dict):
    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#convert_script:{prov_dict["eid"]}', properties={
        'endTime': prov_dict['end_time'],
        'name': 'WSI Tiler'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    instrument = crate.add(HistopatScript(crate, prov_dict['git_commit_hash'], prov_dict['script'], properties={'name': 'WSI Tiler Python Script'}))
    ce_convert.instrument = instrument


    # Create and Map Input WSI Directory Entities
    for k,v in config_dict['slide-converter'].items():
        if k == '_global':
            continue
        cents = process_group(crate, k, v, config_dict['slide-converter']['_global']['pattern'])
        top_dir = crate.add_dataset(dest_path=f'wsi/{k}')
        top_dir['hasPart'] = cents
        ce_convert['object'] += [top_dir]

    # Create and Map Input Configuration File Entity
    ce_convert['object'] += [crate.add_file(prov_dict['config_file'], properties={
        'name': 'input configuration file',
        'encodingFormat': 'text/json'
    })]

    # Create and Map Output File Entities
    prov_log = crate.add_file(str(log_fp), properties={
        'name': 'output provenance log',
        'encodingFormat': 'application/json'
    })
    ce_convert['result'] += [
        # Output dataset
        crate.add_file(prov_dict['dataset_file'], properties={
            'name': 'output dataset',
            'encodingFormat': 'application/x-hdf5'
        }),
        # Output provenance log
        prov_log
    ]

    # First CreateAction Entity
    ce_convert = crate.add(HistopatEntity(crate, f'#convert_script:{meta_prov_dict["eid"]}:CPM-provgen', properties={
        'name': 'CPM Compliant Tiler Provenanace Generation Execution',
        'description': 'CPM compliant provenance generation for WSI tiling.'
    }))
    crate.root_dataset['mentions'] += [ce_convert]

    # Create and Map Instrument Entity
    assert Path(meta_prov_dict['script']).exists(), 'preproc provgen script does not exist'
    instrument = crate.add_file(meta_prov_dict['script'], properties={
        'name': 'WSI Tiler Provenanace Generation Python Script',
        '@type': ['File', 'SoftwareSourceCode'],
        'description': 'A python script that translates the computation log files into CPM compliant provenance file.',
    })
    ce_convert.instrument = instrument

    ce_convert['object'] += [prov_log]

    # Create and Map Output Entities
    assert Path(meta_prov_dict['output']['provn']).exists(), 'preproc provn does not exist'
    provn_entity = crate.add_file(meta_prov_dict['output']['provn'], properties={
        '@type': ['File', 'CPMProvenanceFile'],
        'name': 'WSI Tiler Provenanace CPM File',
        'encodingFormat': ['text/provenance-notation', {'@id': 'http://www.w3.org/TR/2013/REC-prov-n-20130430/'}],
        'description': 'CPM compliant provenance file generated based on the computation log file.',
        'about': []
    })
    
    assert Path(meta_prov_dict['output']['png']).exists(), 'preproc PNG provn does not exist'
    provn_png_entity = crate.add_file(meta_prov_dict['output']['png'], properties={
        '@type': ['File'],
        'name': 'PNG visualization of Provenanace CPM File',
        'encodingFormat': 'image/png',
        'description': 'PNG visualization of a CPM compliant provenance file generated based on the computation log file.',
        'about': []
    })
    
    assert meta_log_fp.exists(), 'preproc meta_log_fp does not exist'
    provn_log_entity = crate.add_file(meta_log_fp, properties={
        '@type': ['File'],
        'name': 'Preprocess provgen log file',
        'encodingFormat': 'text/json',
        'description': 'Log file for provenance generation.',
        'about': []
    })
    ce_convert['result'] += [provn_entity, provn_png_entity, provn_log_entity]
    provn_entity['about'] += [ce_convert]
    provn_log_entity['about'] += [ce_convert]
    provn_png_entity['about'] += [ce_convert]

    return crate

def rocrate_preproc(crate, log_fp, meta_log_fp):
    with log_fp.open('r') as json_in:
        prov_dict = json.load(json_in)
        
    with meta_log_fp.open('r') as json_in:
        meta_prov_dict = json.load(json_in)
    
    with Path(prov_dict['config_file']).resolve().open('r') as config_in:
        config_dict = json.load(config_in)
        
    crate = rocrate_module(crate, log_fp, meta_log_fp, prov_dict, meta_prov_dict, config_dict)
    return crate