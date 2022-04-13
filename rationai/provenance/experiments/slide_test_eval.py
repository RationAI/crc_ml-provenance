# Standard Imports
import argparse
from pathlib import Path


# Third-party Imports


# Local Imports
from rationai.provenance import NAMESPACE_COMMON_MODEL
from rationai.provenance import NAMESPACE_PREPROC
from rationai.provenance import NAMESPACE_DCT
from rationai.provenance import NAMESPACE_PROV
from rationai.provenance import NAMESPACE_EVAL
from rationai.provenance import NAMESPACE_TRAINING
from rationai.provenance import prepare_document

from rationai.utils.provenance import parse_log
from rationai.utils.provenance import export_to_image
from rationai.utils.provenance import get_sha256
from rationai.utils.provenance import flatten_dict


def export_provenance(test_log_fp: Path, eval_log_fp: Path) -> None:
    doc = prepare_document()
    log_t = parse_log(test_log_fp)
    log_e = parse_log(eval_log_fp)

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
        f"{NAMESPACE_DCT}:hasPart": f"{NAMESPACE_EVAL}:predict",
        f"{NAMESPACE_DCT}:hasPart2": f"{NAMESPACE_EVAL}:evaluate",
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

    evalConfig = bndl.entity(f'{NAMESPACE_EVAL}:cfgEval', other_attributes={
        'filepath': log_e['config_file'],
        'sha256': get_sha256(log_e['config_file'])
    })

    # Datasets
    testDataset = bndl.entity(f"{NAMESPACE_EVAL}:testHDF5Dataset", other_attributes=log_t['splits']['test_gen'])
    evalDataset = bndl.entity(f"{NAMESPACE_EVAL}:evalHDF5Dataset", other_attributes=log_e['splits']['eval_gen'])

    # Main Activities
    testRun = bndl.activity(f"{NAMESPACE_EVAL}:predict", other_attributes={
        'experiment_ID': log_t['eid'],
        f"git_commit_hash": log_t['git_commit_hash']
    })

    evalRun = bndl.activity(f'{NAMESPACE_EVAL}:evaluate', other_attributes={
        'experiment_ID': log_e['eid'],
        f"git_commit_hash": log_e['git_commit_hash'],
    })

    # Data Entities
    predict_file_attrs = {'filepath': log_t['predictions']['prediction_file'], 'sha256': get_sha256(log_t['predictions']['prediction_file'])}
    predict_table_hashes = log_t['predictions']['sha256']
    testPredicts = bndl.entity(f"{NAMESPACE_EVAL}:testPredictions", other_attributes= predict_file_attrs | predict_table_hashes)
    testEvals = bndl.entity(f"{NAMESPACE_EVAL}:testEvaluations", other_attributes=flatten_dict(log_e['eval']))

    # Input Data Specializations
    bndl.specialization(trainedNetSpec, trainedNet)
    bndl.specialization(testDataset, bckbTestDatset)

    # Result Derivations
    bndl.wasDerivedFrom(testPredicts, testDataset)
    bndl.wasDerivedFrom(testEvals, evalDataset)

    # PREDICT Activity Relationships
    bndl.used(testRun, testDataset)
    bndl.used(testRun, testConfig)
    bndl.used(testRun, trainedNetSpec)
    bndl.wasGeneratedBy(testPredicts, testRun)

    # Read Data
    bndl.wasDerivedFrom(evalDataset, testPredicts)

    # EVAL Activity Relationships
    bndl.used(evalRun, evalDataset)
    bndl.used(evalRun, evalConfig)
    bndl.wasGeneratedBy(testEvals, evalRun)

    export_to_image(bndl, 'evaluation')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--test_log_fp', type=Path, required=True, help='Path to provenanace log of a test run')
    parser.add_argument('--eval_log_fp', type=Path, required=True, help='Path to provenanace log of an eval run')
    args = parser.parse_args()

    export_provenance(args.test_log_fp, args.eval_log_fp)