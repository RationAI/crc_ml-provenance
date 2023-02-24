# Standard Imports
import argparse
from pathlib import Path
import json
import uuid
import os
import pygit2

# Local Imports
from rationai.provenance import NAMESPACE_COMMON_MODEL
from rationai.provenance import NAMESPACE_PREPROC
from rationai.provenance import NAMESPACE_DCT
from rationai.provenance import NAMESPACE_PROV
from rationai.provenance import NAMESPACE_EVAL
from rationai.provenance import NAMESPACE_TRAINING
from rationai.provenance import prepare_document

from rationai.utils.provenance import parse_log
from rationai.utils.provenance import export_to_image, export_to_provn
from rationai.utils.provenance import get_sha256
from rationai.utils.provenance import flatten_dict


def export_provenance(experiment_dir: Path) -> None:
    log_fp =  experiment_dir / 'prov_test.log'
    assert log_fp.exists(), 'Execution log not found.'
    
    doc = prepare_document()
    log_t = parse_log(log_fp)

    bndl = doc.bundle(f"{NAMESPACE_EVAL}:bundle_eval")

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###
    
  
    #creating connectors
    connEntTrainedNet = bndl.entity(f"{NAMESPACE_TRAINING}:trainedModelConnector", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverConnector",
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": f"{NAMESPACE_TRAINING}:bundle_training",
        f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": f"#URI#"
    })

    connEntTestSet = bndl.entity(f"{NAMESPACE_PREPROC}:datasetTestConnector", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverConnector",
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": f"{NAMESPACE_PREPROC}:bundle_preproc",
        f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": f"#URI#"
    })

    # creating agents
    sendAgentTrainedNet = bndl.agent(f"{NAMESPACE_EVAL}:senderAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderAgent"
    })

    sendAgentTestSet = bndl.agent(f"{NAMESPACE_EVAL}:senderAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderAgent"
    })

    #creating external inputs
    trainedNet = bndl.entity(f"{NAMESPACE_EVAL}:trainedNet", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:externalInput"
    })
    bckbTestDatset = bndl.entity(f"{NAMESPACE_EVAL}:testDataset", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:externalInput"
    })

    #creating receipt activities
    rec = bndl.activity(f"{NAMESPACE_EVAL}:receipt1", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })
    rec2 = bndl.activity(f"{NAMESPACE_EVAL}:receipt2", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })

    # creating main activity
    mainActivityTesting = bndl.activity(f"{NAMESPACE_EVAL}:mainActivityTesting", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:mainActivity",
        f"{NAMESPACE_DCT}:hasPart": f"{NAMESPACE_EVAL}:test"
    })

    #backbone relations
    bndl.wasDerivedFrom(trainedNet, connEntTrainedNet)
    bndl.wasInvalidatedBy(connEntTrainedNet, rec)
    bndl.wasGeneratedBy(trainedNet, rec)
    bndl.used(rec, connEntTrainedNet)

    bndl.wasDerivedFrom(bckbTestDatset, connEntTestSet)
    bndl.wasGeneratedBy(bckbTestDatset, rec2)
    bndl.used(rec2, connEntTestSet)
    bndl.wasInvalidatedBy(connEntTestSet, rec2)

    bndl.used(mainActivityTesting, bckbTestDatset)
    bndl.used(mainActivityTesting, trainedNet)

    bndl.attribution(connEntTrainedNet, sendAgentTrainedNet)
    bndl.attribution(connEntTestSet, sendAgentTestSet)


    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    trainedNetSpec = bndl.entity(f"{NAMESPACE_EVAL}:trainedNetFromCkpt", other_attributes=log_t['model'])
    
    # Configs
    testConfig = bndl.entity(f'{NAMESPACE_EVAL}:cfgTest', other_attributes={
        'filepath': log_t['config_file'],
        'sha256': get_sha256(log_t['config_file'])
    })

    predict_file_attrs = {'filepath': log_t['predictions']['prediction_file'], 'sha256': get_sha256(log_t['predictions']['prediction_file'])}
    
    # Datasets
    testDataset = bndl.entity(f"{NAMESPACE_EVAL}:testHDF5Dataset", other_attributes=log_t['splits']['test_gen'])
    evalDataset = bndl.entity(f"{NAMESPACE_EVAL}:evalHDF5Dataset", other_attributes=predict_file_attrs)

    # Main Activities
    testRun = bndl.activity(f"{NAMESPACE_EVAL}:test", other_attributes={
        'experiment_ID': log_t['eid'],
        f"git_commit_hash": log_t['git_commit_hash']
    })

    # Data Entities
    predict_table_hashes = log_t['predictions']['sha256']
    predict_table_hashes.update(flatten_dict(log_t['evaluations']))
    testPredicts = bndl.entity(f"{NAMESPACE_EVAL}:testPredictions", other_attributes=predict_table_hashes)

    # Input Data Specializations
    bndl.specialization(trainedNetSpec, trainedNet)
    bndl.specialization(testDataset, bckbTestDatset)

    # Result Derivations
    bndl.wasDerivedFrom(testPredicts, testDataset)

    # PREDICT Activity Relationships
    bndl.used(testRun, testDataset)
    bndl.used(testRun, testConfig)
    bndl.used(testRun, trainedNetSpec)
    bndl.wasGeneratedBy(testPredicts, testRun)

    # Read Data
    bndl.wasDerivedFrom(evalDataset, testPredicts)

    export_to_image(bndl, (experiment_dir / log_fp.stem).with_suffix('.png'))
    export_to_provn(doc, (experiment_dir / log_fp.stem).with_suffix('.provn'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to provenanace log of a test run')
    parser.add_argument('--eid', type=str, required=True, help='Execution UUID')
    args = parser.parse_args()
    
    with open(args.config_fp, 'r') as json_in:
        json_cfg = json.load(json_in)
    
    experiment_dir = Path(json_cfg['output_dir']) / args.eid
    
    # Provenance of provenance
    output_log = {
        'git_commit_hash': str(pygit2.Repository('.').revparse_single('HEAD').hex),
        'script': str(__file__),
        'eid': str(uuid.uuid4()),
        'input': str(args.config_fp.resolve()),
        'output': {
            'png': str(experiment_dir / 'prov_test.png'),
            'provn': str(experiment_dir / 'prov_test.provn')
        }
    }
    with open(experiment_dir / 'prov_test.provn.log', 'w') as json_out:
        json.dump(output_log, json_out, indent=3)
    
    export_provenance(experiment_dir)