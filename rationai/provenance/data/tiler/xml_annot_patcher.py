# Standard Imports
import argparse
from pathlib import Path
import json
import uuid
import os


# Third-party Imports


# Local Imports
from rationai.provenance import NAMESPACE_COMMON_MODEL
from rationai.provenance import NAMESPACE_PATHOLOGY
from rationai.provenance import NAMESPACE_PREPROC
from rationai.provenance import NAMESPACE_DCT
from rationai.provenance import NAMESPACE_PROV
from rationai.provenance import NAMESPACE_EVAL
from rationai.provenance import NAMESPACE_TRAINING
from rationai.provenance import prepare_document

from rationai.utils.provenance import parse_log
from rationai.utils.provenance import export_to_image, export_to_provn
from rationai.utils.provenance import flatten_lists
from rationai.utils.provenance import get_sha256
from rationai.utils.provenance import hash_tables_by_groups


def export_provenance(experiment_dir: Path) -> None:
    log_fp =  experiment_dir / 'prov_preprocess.log'
    assert log_fp.exists(), 'Execution log not found.'
    
    doc = prepare_document()
    log_t = parse_log(log_fp)
    # Creating preprocessing bundle
    bndl = doc.bundle(f"{NAMESPACE_PREPROC}:bundle_preproc")

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###

    # Receiver connector
    recConnEnt = bndl.entity(f"{NAMESPACE_PATHOLOGY}:WSIDataConnector", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverConnector",
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": f"{NAMESPACE_PATHOLOGY}:bundle_pathology",
        f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": "#URI#"
    })

    # Sender connectors
    sendTrainingConnEntDataset = bndl.entity(f"{NAMESPACE_PREPROC}:datasetTrainConnector", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderConnector",
        f"{NAMESPACE_COMMON_MODEL}:receiverBundleId": f"{NAMESPACE_TRAINING}:bundle_training",
        f"{NAMESPACE_COMMON_MODEL}:receiverServiceUri": f"#URI#"
    })

    sendTestingConnEntDataset = bndl.entity(f"{NAMESPACE_PREPROC}:datasetTestConnector", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderConnector",
        f"{NAMESPACE_COMMON_MODEL}:receiverBundleId": f"{NAMESPACE_EVAL}:bundle_eval",
        f"{NAMESPACE_COMMON_MODEL}:receiverServiceUri": f"#URI#"
    })

    # Receiver agent
    recAgent = bndl.agent(f"{NAMESPACE_PREPROC}:receiverAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverAgent"
    })

    # Sending agent
    sendAgent = bndl.agent(f"{NAMESPACE_PREPROC}:senderAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderAgent"
    })

    # Receipt activity
    rec = bndl.activity(f"{NAMESPACE_PREPROC}:receipt", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })

    # Main activity
    preproc = bndl.activity(f"{NAMESPACE_PREPROC}:preprocessing", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:mainActivity",
        f"{NAMESPACE_DCT}:hasPart": f"{NAMESPACE_PREPROC}:tilesGeneration",
    })

    # Data Entity Node
    rawDataEnt = bndl.entity(f"{NAMESPACE_PREPROC}:WSI data", other_attributes={
        f"{NAMESPACE_COMMON_MODEL}:primaryId": f""
    })

    # Establish relationships between backbones nodes
    bndl.wasGeneratedBy(rawDataEnt, rec)
    bndl.attribution(recConnEnt, sendAgent)
    bndl.attribution(sendTrainingConnEntDataset, recAgent)
    bndl.attribution(sendTestingConnEntDataset, recAgent)
    bndl.wasDerivedFrom(rawDataEnt, recConnEnt)
    bndl.used(rec, recConnEnt)
    bndl.wasInvalidatedBy(recConnEnt, rec)

    bndl.used(preproc, rawDataEnt)
    bndl.wasDerivedFrom(sendTrainingConnEntDataset, rawDataEnt)
    bndl.wasGeneratedBy(sendTrainingConnEntDataset, preproc)
    bndl.wasDerivedFrom(sendTestingConnEntDataset, rawDataEnt)
    bndl.wasGeneratedBy(sendTestingConnEntDataset, preproc)


    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    # Activity Node
    gzact = bndl.activity(f"{NAMESPACE_PREPROC}:tilesGeneration", other_attributes={
        f"{NAMESPACE_PROV}:label": f"tiles generation",
        "git_commit_hash": log_t['git_commit_hash']
    })

    # Output Entity Node
    hdf_file = bndl.entity(f'{NAMESPACE_PREPROC}:hdf5_dataset', other_attributes={
        'filepath': log_t['dataset_file'],
        'hash': get_sha256(log_t['dataset_file'])
    })

    RAW_DATA_SPECS = []
    cfg = log_t['config']['slide-converter']

    # Global Config Node
    global_cfg = flatten_lists(cfg.pop('_global'))
    cfg_global = bndl.entity(f"{NAMESPACE_PREPROC}:params", other_attributes={
        "filepath": log_t['config_file'],
        "sha256": get_sha256(log_t['config_file'])
    })

    # Group Config Nodes
    table_hashes = hash_tables_by_groups(log_t['dataset_file'], cfg.keys())
    for group_name, group_itemlist in cfg.items():
        hdf5_group = bndl.entity(f"{NAMESPACE_PREPROC}:{group_name}Group", other_attributes=table_hashes[group_name])
        for data_folder in group_itemlist:
            # Folder Data Entity Node
            rawDataSpec = bndl.entity(f"{NAMESPACE_PREPROC}:Data {Path(data_folder['slide_dir']).name}", other_attributes={
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
            rawDataCfg = bndl.entity(f"{NAMESPACE_PREPROC}:Config {Path(data_folder['slide_dir']).name}", other_attributes=(_config))

            # Folder Table Entity Node
            roiDataTable = bndl.entity(f"{NAMESPACE_PREPROC}:roiTables {Path(data_folder['slide_dir']).name}", other_attributes={})

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
    bndl.specialization(hdf_file, sendTestingConnEntDataset)

    export_to_image(bndl, (experiment_dir / log_fp.stem).with_suffix('.png'))
    export_to_provn(doc, (experiment_dir / log_fp.stem).with_suffix('.provn'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to provenanace log of a WSI conversion run')
    args = parser.parse_args()
    
    with open(args.config_fp, 'r') as json_in:
        json_cfg = json.load(json_in)
    
    experiment_dir = Path(json_cfg['slide-converter']['_global']['output_dir'])
    
    # Provenance of provenance
    output_log = {
        'script': str(__file__),
        'eid': str(uuid.uuid4()),
        'input': str(args.config_fp.resolve()),
        'output': {
            'png': str(experiment_dir / 'prov_preprocess.png'),
            'provn': str(experiment_dir / 'prov_preprocess.provn')
        }
    }
    with open(experiment_dir / 'prov_preprocess.provn.log', 'w') as json_out:
        json.dump(output_log, json_out, indent=3)
    
    export_provenance(experiment_dir)
