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
from rationai.provenance import BUNDLE_PREPROC
from rationai.provenance import BUNDLE_TRAIN
from rationai.provenance import BUNDLE_META
from rationai.provenance import BUNDLE_EVAL
from rationai.provenance import OUTPUT_DIR
from rationai.provenance import ORGANISATION_DOI

# Document Namespaces
from rationai.provenance import DEFAULT_NAMESPACE_URI
from rationai.provenance import PROVN_NAMESPACE
from rationai.provenance import PROVN_NAMESPACE_URI
from rationai.provenance import DOI_NAMESPACE
from rationai.provenance import DOI_NAMESPACE_URI
from rationai.provenance import NAMESPACE_COMMON_MODEL
from rationai.provenance import NAMESPACE_DCT
from rationai.provenance import NAMESPACE_PROV
from rationai.provenance import GRAPH_NAMESPACE_URI

from rationai.utils.provenance import parse_log
from rationai.utils.provenance import export_to_image
from rationai.utils.provenance import export_to_file
from rationai.utils.provenance import flatten_lists
from rationai.utils.provenance import get_sha256
from rationai.utils.provenance import hash_tables_by_groups


def prepare_document():
    document = prov.ProvDocument()
    
    # Declaring namespaces
    document.add_namespace(PROVN_NAMESPACE, PROVN_NAMESPACE_URI)
    document.add_namespace(DOI_NAMESPACE, DOI_NAMESPACE_URI)
    document.add_namespace(NAMESPACE_COMMON_MODEL, "cpm_uri")
    document.add_namespace(NAMESPACE_DCT, "dct_uri")
    document.set_default_namespace(DEFAULT_NAMESPACE_URI)

    return document


def export_provenance(config_fp: Path) -> None:
    with open(config_fp, 'r') as json_in:
        json_cfg = json.load(json_in)
    
    experiment_dir = Path(json_cfg['slide-converter']['_global']['output_dir'])
    log_fp =  (experiment_dir / BUNDLE_PREPROC).with_suffix('.log')
    assert log_fp.exists(), 'Execution log not found.'
    
    doc = prepare_document()
    log_t = parse_log(log_fp)
    # Creating preprocessing bundle
    bndl = doc.bundle(f'{PROVN_NAMESPACE}:{BUNDLE_PREPROC}')

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###

    # Sender connectors
    entity_identifier = 'datasetTrainConnector'
    sendTrainingConnEntDataset = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderConnector",
        f"{NAMESPACE_COMMON_MODEL}:receiverBundleId": f'{PROVN_NAMESPACE}:{BUNDLE_TRAIN}',
        f"{NAMESPACE_COMMON_MODEL}:receiverServiceUri": f'{DEFAULT_NAMESPACE_URI}',
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}'
    })

    entity_identifier = 'datasetEvalConnector'
    sendEvalConnEntDataset = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderConnector",
        f"{NAMESPACE_COMMON_MODEL}:receiverBundleId": f"{PROVN_NAMESPACE}:{BUNDLE_EVAL}",
        f"{NAMESPACE_COMMON_MODEL}:receiverServiceUri": f'{DEFAULT_NAMESPACE_URI}',
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}'
    })
    
    # External Input Connector
    entity_identifier = 'WSIDataExternalInputConnector'
    rawDataEnt = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:externalInputConnector",
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}',
        f'{NAMESPACE_COMMON_MODEL}:currentBundle': str(bndl.identifier)
    })
    
    # Receiver agent
    recAgent = bndl.agent(f"receiverAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverAgent"
    })

    # Main activity
    preproc = bndl.activity(f"preprocessing", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:mainActivity",
        f"{NAMESPACE_DCT}:hasPart": f"tilesGeneration",
    })
    
    # Receipt Activity
    act_receipt = bndl.activity(f"receipt", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })
    

    # Establish relationships between backbones nodes
    bndl.wasGeneratedBy(rawDataEnt, act_receipt)
    bndl.attribution(sendTrainingConnEntDataset, recAgent)
    bndl.attribution(sendEvalConnEntDataset, recAgent)

    bndl.used(preproc, rawDataEnt)
    bndl.wasDerivedFrom(sendTrainingConnEntDataset, rawDataEnt)
    bndl.wasGeneratedBy(sendTrainingConnEntDataset, preproc)
    bndl.wasDerivedFrom(sendEvalConnEntDataset, rawDataEnt)
    bndl.wasGeneratedBy(sendEvalConnEntDataset, preproc)


    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    # Activity Node
    gzact = bndl.activity(f"tilesGeneration", other_attributes={
        f"{NAMESPACE_PROV}:label": f"tiles generation",
        "git_commit_hash": log_t['git_commit_hash']
    })

    # Output Entity Node
    hdf_file = bndl.entity(f'hdf5_dataset', other_attributes={
        'filepath': log_t['dataset_file'],
        'hash': get_sha256(log_t['dataset_file'])
    })

    RAW_DATA_SPECS = []
    cfg = log_t['config']['slide-converter']

    # Global Config Node
    global_cfg = flatten_lists(cfg.pop('_global'))
    cfg_global = bndl.entity(f"params", other_attributes={
        "filepath": log_t['config_file'],
        "sha256": get_sha256(log_t['config_file'])
    })

    # Group Config Nodes
    table_hashes = hash_tables_by_groups(log_t['dataset_file'], cfg.keys())
    for group_name, group_itemlist in cfg.items():
        hdf5_group = bndl.entity(f"{group_name}Group", other_attributes=table_hashes[group_name])
        for data_folder in group_itemlist:
            # Folder Data Entity Node
            rawDataSpec = bndl.entity(f"Data_{Path(data_folder['slide_dir']).name}", other_attributes={
                f"{NAMESPACE_COMMON_MODEL}:primaryId": f"",
                f"imagesDirSHA256": f"{get_sha256(data_folder['slide_dir'])}",
                f"imagesDirPath": f"{data_folder['slide_dir']}",
                f"annotationsDirHash": f"{get_sha256(data_folder['label_dir'])}",
                f"annotationsDirPath": f"{data_folder['label_dir']}",
                f"{NAMESPACE_PROV}:type": f"{NAMESPACE_PROV}:collection"
            })

            # Folder Config Entity Node
            data_folder = flatten_lists(data_folder)
            _config = dict(global_cfg)
            _config.update(data_folder)
            rawDataCfg = bndl.entity(f"Config_{Path(data_folder['slide_dir']).name}", other_attributes=(_config))

            # Folder Table Entity Node
            roiDataTable = bndl.entity(f"roiTables_{Path(data_folder['slide_dir']).name}", other_attributes={})

            # Establish relationships between dynamic nodes
            bndl.wasDerivedFrom(rawDataCfg, cfg_global)     # [Global Config] -WDF-> [Folder Config]
            bndl.wasGeneratedBy(roiDataTable, gzact)        # [Folder Table] -WGB-> [Activity]
            bndl.wasDerivedFrom(roiDataTable, rawDataSpec)  # [Folder Table] -WDF-> [Folder Data]
            bndl.wasDerivedFrom(roiDataTable, rawDataCfg)   # [Folder Table] -WDF-> [Folder Config]
            bndl.used(gzact, rawDataSpec)                   # [Activity] -U-> [Folder Data]
            bndl.used(gzact, rawDataCfg)                    # [Activity] -U-> [Folder Config]
            bndl.wasDerivedFrom(hdf5_group, roiDataTable)   # [Group Table] -WDF-> [Folder Table]
            bndl.specialization(rawDataSpec, rawDataEnt)    # [Folder Data] -S-> [DataConnectorEntity]
        bndl.wasDerivedFrom(hdf_file, hdf5_group)

    # Relationships between static parts
    bndl.specialization(hdf_file, sendTrainingConnEntDataset)
    bndl.specialization(hdf_file, sendEvalConnEntDataset)
    
    subdirs = ['', 'graph', 'json', 'provn']
    for subdir in subdirs:
        if not (OUTPUT_DIR / subdir).exists():
            (OUTPUT_DIR / subdir).mkdir(parents=True)

    export_to_image(bndl, (OUTPUT_DIR / 'graph' / BUNDLE_PREPROC).with_suffix('.png'))
    export_to_file(doc, (OUTPUT_DIR / 'json' / BUNDLE_PREPROC).with_suffix('.json'), format='json')
    export_to_file(doc, OUTPUT_DIR / 'provn' / BUNDLE_PREPROC, format='provn')
    
    # Provenance of provenance
    output_log = {
        'git_commit_hash': str(pygit2.Repository('.').revparse_single('HEAD').hex),
        'script': str(__file__),
        'eid': str(uuid.uuid4()),
        'input': {
            'config': str(config_fp.resolve()),
            'log': str(log_fp.resolve())
        },
        'output': {
            'local_png': str((OUTPUT_DIR / 'graph' / BUNDLE_PREPROC).with_suffix('.png')),
            'remote_png': str(GRAPH_NAMESPACE_URI + str(Path(BUNDLE_PREPROC).with_suffix('.png'))),
            'local_provn': str(OUTPUT_DIR / 'provn' / BUNDLE_PREPROC),
            'remote_provn': str(PROVN_NAMESPACE_URI + BUNDLE_PREPROC)
        }
    }
    
    with open(experiment_dir / f'{BUNDLE_PREPROC}.log', 'w') as json_out:
        json.dump(output_log, json_out, indent=3)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to provenanace log of a WSI conversion run')
    args = parser.parse_args()
    
    export_provenance(args.config_fp)
