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
from rationai.utils.provenance import get_sha256
from rationai.utils.provenance import flatten_dict


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
    
    experiment_dir = Path(json_cfg['output_dir']) / args.eid
    
    log_fp =  (experiment_dir / BUNDLE_EVAL).with_suffix('.log')
    assert log_fp.exists(), 'Execution log not found.'
    
    doc = prepare_document()
    log_t = parse_log(log_fp)

    bndl = doc.bundle(f"{PROVN_NAMESPACE}:{BUNDLE_EVAL}")

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###
    
  
    #creating connectors
    entity_identifier = 'trainedModelConnector'
    connEntTrainedNet = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverConnector",
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": f"{PROVN_NAMESPACE}:{BUNDLE_TRAIN}",
        f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": f'{DEFAULT_NAMESPACE_URI}',
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}'
    })

    entity_identifier = 'datasetEvalConnector'
    connEntEvalSet = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverConnector",
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": f"{PROVN_NAMESPACE}:{BUNDLE_PREPROC}",
        f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": f'{DEFAULT_NAMESPACE_URI}',
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}'
    })

    # creating agents
    sendAgentTrainedNet = bndl.agent(f"senderAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderAgent"
    })

    sendAgentEvalSet = bndl.agent(f"senderAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderAgent"
    })

    # External Input Connectors
    entity_identifier = 'trainedNetExternalInputConnector'
    trainedNet = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:externalInputConnector",
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}',
        f'{NAMESPACE_COMMON_MODEL}:currentBundle': str(bndl.identifier)
    })
    
    entity_identifier = 'testDatasetExternalInputConnector'
    bckbEvalDatset = bndl.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:externalInputConnector",
        f"{NAMESPACE_COMMON_MODEL}:metabundle": f'{PROVN_NAMESPACE}:{BUNDLE_META}',
        f'{NAMESPACE_COMMON_MODEL}:currentBundle': str(bndl.identifier)
    })

    #creating receipt activities
    rec = bndl.activity(f"receipt1", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })
    rec2 = bndl.activity(f"receipt2", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })

    # creating main activity
    mainActivityEval = bndl.activity(f"mainActivityEval", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:mainActivity",
        f"{NAMESPACE_DCT}:hasPart": f"test"
    })

    #backbone relations
    bndl.wasDerivedFrom(trainedNet, connEntTrainedNet)
    bndl.wasInvalidatedBy(connEntTrainedNet, rec)
    bndl.wasGeneratedBy(trainedNet, rec)
    bndl.used(rec, connEntTrainedNet)

    bndl.wasDerivedFrom(bckbEvalDatset, connEntEvalSet)
    bndl.wasGeneratedBy(bckbEvalDatset, rec2)
    bndl.used(rec2, connEntEvalSet)
    bndl.wasInvalidatedBy(connEntEvalSet, rec2)

    bndl.used(mainActivityEval, bckbEvalDatset)
    bndl.used(mainActivityEval, trainedNet)

    bndl.attribution(connEntTrainedNet, sendAgentTrainedNet)
    bndl.attribution(connEntEvalSet, sendAgentEvalSet)


    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    trainedNetSpec = bndl.entity(f"trainedNetFromCkpt", other_attributes=log_t['model'])
    
    # Configs
    testConfig = bndl.entity(f'cfgEval', other_attributes={
        'filepath': log_t['config_file'],
        'sha256': get_sha256(log_t['config_file'])
    })

    predict_file_attrs = {'filepath': log_t['predictions']['prediction_file'], 'sha256': get_sha256(log_t['predictions']['prediction_file'])}
    
    # Datasets
    testDataset = bndl.entity(f"testHDF5Dataset", other_attributes=log_t['splits']['test_gen'])
    evalDataset = bndl.entity(f"evalHDF5Dataset", other_attributes=predict_file_attrs)

    # Main Activities
    testRun = bndl.activity(f"test", other_attributes={
        'experiment_ID': log_t['eid'],
        f"git_commit_hash": log_t['git_commit_hash']
    })

    # Data Entities
    predict_table_hashes = log_t['predictions']['sha256']
    predict_table_hashes.update(flatten_dict(log_t['evaluations']))
    testPredicts = bndl.entity(f"testPredictions", other_attributes=predict_table_hashes)

    # Input Data Specializations
    bndl.specialization(trainedNetSpec, trainedNet)
    bndl.specialization(testDataset, bckbEvalDatset)

    # Result Derivations
    bndl.wasDerivedFrom(testPredicts, testDataset)

    # PREDICT Activity Relationships
    bndl.used(testRun, testDataset)
    bndl.used(testRun, testConfig)
    bndl.used(testRun, trainedNetSpec)
    bndl.wasGeneratedBy(testPredicts, testRun)

    # Read Data
    bndl.wasDerivedFrom(evalDataset, testPredicts)

    subdirs = ['', 'graph', 'json', 'provn']
    for subdir in subdirs:
        if not (OUTPUT_DIR / subdir).exists():
            (OUTPUT_DIR / subdir).mkdir(parents=True)
    
    export_to_image(bndl, (OUTPUT_DIR / 'graph' / BUNDLE_EVAL).with_suffix('.png'))
    export_to_file(doc, (OUTPUT_DIR / 'json' / BUNDLE_EVAL).with_suffix('.json'), format='json')
    export_to_file(doc, OUTPUT_DIR / 'provn' / BUNDLE_EVAL, format='provn')
    
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
            'local_png': str((OUTPUT_DIR / 'graph' / BUNDLE_EVAL).with_suffix('.png')),
            'remote_png': str(GRAPH_NAMESPACE_URI + str(Path(BUNDLE_EVAL).with_suffix('.png'))),
            'local_provn': str(OUTPUT_DIR / 'provn' / BUNDLE_EVAL),
            'remote_provn': str(PROVN_NAMESPACE_URI + BUNDLE_EVAL)
        }
    }
    with open(experiment_dir / f'{BUNDLE_EVAL}.log', 'w') as json_out:
        json.dump(output_log, json_out, indent=3)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to provenanace log of a test run')
    parser.add_argument('--eid', type=str, required=True, help='Execution UUID')
    args = parser.parse_args()
   
    export_provenance(args.config_fp)