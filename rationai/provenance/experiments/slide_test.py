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
from rationai.provenance import NAMESPACE_COMMON_MODEL_URI
from rationai.provenance import NAMESPACE_DCT
from rationai.provenance import NAMESPACE_DCT_URI
from rationai.provenance import NAMESPACE_PROV
from rationai.provenance import GRAPH_NAMESPACE_URI
from rationai.provenance import BACKWARD_PROVN_NAMESPACE
from rationai.provenance import BACKWARD_PROVN_NAMESPACE_URI
from rationai.provenance import BACKWARD_BUNDLE

from rationai.utils.provenance import parse_log
from rationai.utils.provenance import export_to_image
from rationai.utils.provenance import export_to_file
from rationai.utils.provenance import get_hash
from rationai.utils.provenance import flatten_dict


def add_namespaces(prov_obj):
    # Declaring namespaces
    prov_obj.add_namespace(PROVN_NAMESPACE, PROVN_NAMESPACE_URI)
    prov_obj.add_namespace(BACKWARD_PROVN_NAMESPACE, BACKWARD_PROVN_NAMESPACE_URI)
    prov_obj.add_namespace(DOI_NAMESPACE, DOI_NAMESPACE_URI)
    prov_obj.add_namespace(NAMESPACE_COMMON_MODEL, NAMESPACE_COMMON_MODEL_URI)
    prov_obj.add_namespace(NAMESPACE_DCT, NAMESPACE_DCT_URI)
    prov_obj.set_default_namespace(DEFAULT_NAMESPACE_URI)


def export_provenance(config_fp: Path) -> None:
    with open(config_fp, 'r') as json_in:
        json_cfg = json.load(json_in)
    
    experiment_dir = Path(json_cfg['output_dir']) / args.eid
    
    log_fp =  (experiment_dir / BUNDLE_EVAL).with_suffix('.log')
    assert log_fp.exists(), 'Execution log not found.'
    
    doc = prov.ProvDocument()
    add_namespaces(doc)
    
    log_t = parse_log(log_fp)

    bndl = doc.bundle(f"{PROVN_NAMESPACE}:{BUNDLE_EVAL}")
    add_namespaces(bndl)

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###
    
  
    # Backward Connectors
    entity_identifier = 'trainedModelConnector'
    connEntTrainedNet = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:backwardConnector"),
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": bndl.valid_qualified_name(f"{PROVN_NAMESPACE}:{BUNDLE_TRAIN}"),
        #f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": bndl.valid_qualified_name(f'{DEFAULT_NAMESPACE_URI}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })

    entity_identifier = 'evalIndexROITablesConnector'
    connEntEvalSet = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:backwardConnector"),
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": bndl.valid_qualified_name(f"{PROVN_NAMESPACE}:{BUNDLE_PREPROC}"),
        #f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": bndl.valid_qualified_name(f'{DEFAULT_NAMESPACE_URI}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
    
    entity_identifier = 'WSIDataConnectorEval'
    connEntWSIData = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:backwardConnector"),
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": bndl.valid_qualified_name(f"{BACKWARD_PROVN_NAMESPACE}:{BACKWARD_BUNDLE}"),
        #f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": bndl.valid_qualified_name(f'{BACKWARD_PROVN_NAMESPACE}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{BACKWARD_PROVN_NAMESPACE}:{BUNDLE_META}'),
    })
    
    # Current Connectors
    entity_identifier = 'trainedModel'
    entTrainedNet = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f'{NAMESPACE_COMMON_MODEL}:currentConnector'),
        f"{NAMESPACE_COMMON_MODEL}:currentBundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_EVAL}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
    
    entity_identifier = 'evalIndexROITables'
    entEvalSet = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f'{NAMESPACE_COMMON_MODEL}:currentConnector'),
        f"{NAMESPACE_COMMON_MODEL}:currentBundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_EVAL}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
    
    entity_identifier = 'WSIDataEval'
    entWSIData = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f'{NAMESPACE_COMMON_MODEL}:currentConnector'),
        f"{NAMESPACE_COMMON_MODEL}:currentBundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_EVAL}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
    
    entity_identifier = 'evalPredictionsWithMetrics'
    entEvalSetWithMetrics = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f'{NAMESPACE_COMMON_MODEL}:currentConnector'),
        f"{NAMESPACE_COMMON_MODEL}:currentBundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_EVAL}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })

    # Sender Agents
    sendAgentRDC = bndl.agent(f"researchDataCentre", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:senderAgent")
    })

    sendAgentMMCI = bndl.agent(f"pathologyDepartment", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:senderAgent")
    })
    

    #creating receipt activities
    rec = bndl.activity(f"receiptTrainedNet", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:receiptActivity"),
    })
    
    rec2 = bndl.activity(f"receiptEvalROITables", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:receiptActivity"),
    })
    
    rec3 = bndl.activity(f"receiptWSIDataset", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:receiptActivity"),
    })

    # creating main activity
    mainActivityEval = bndl.activity(f"predictAndEvaluate", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:mainActivity"),
        f"{NAMESPACE_DCT}:hasPart": f"test"
    })

    # BACKBONE RELATIONS
    
    # Agent-Connector Attributed
    bndl.wasAttributedTo(connEntTrainedNet, sendAgentRDC)
    bndl.wasAttributedTo(connEntEvalSet, sendAgentRDC)
    bndl.wasAttributedTo(connEntWSIData, sendAgentMMCI)
    
    # Connector-Receipt Used/Ivalidate
    bndl.wasInvalidatedBy(connEntTrainedNet, rec)
    bndl.wasInvalidatedBy(connEntEvalSet, rec2)
    bndl.wasInvalidatedBy(connEntWSIData, rec3)
    bndl.used(rec, connEntTrainedNet)
    bndl.used(rec2, connEntEvalSet)
    bndl.used(rec3, connEntWSIData)
    
    # Connector Derivations
    bndl.wasDerivedFrom(entTrainedNet, connEntTrainedNet)
    bndl.wasDerivedFrom(entEvalSet, connEntEvalSet)
    bndl.wasDerivedFrom(entWSIData, connEntWSIData)
    
    
    # Agent Generated
    bndl.wasGeneratedBy(entTrainedNet, rec)
    bndl.wasGeneratedBy(entEvalSet, rec2)
    bndl.wasGeneratedBy(entWSIData, rec3)
    
    # Main Activity Usage
    bndl.used(mainActivityEval, entTrainedNet)
    bndl.used(mainActivityEval, entEvalSet)
    bndl.used(mainActivityEval, entWSIData)
    bndl.wasGeneratedBy(entEvalSetWithMetrics, mainActivityEval)
    
    # Output derivations
    bndl.wasDerivedFrom(entEvalSetWithMetrics, entEvalSet)

    
    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    # Domain-specific Entities
    entTrainedNetSpec = bndl.entity(f"VGG16ProstateClassifier", other_attributes=log_t['model'])
    
    # Datasets
    entTestDataset = bndl.entity(f"evalIndexROITablesProstate", other_attributes={
        'filepath': log_t['predictions']['prediction_file'],
        'sha256': get_hash(log_t['predictions']['prediction_file'], hash_type='sha256')
    } | log_t['splits']['test_gen'])
    
    entWSITestData = bndl.entity("WSIDataEvalProstate", other_attributes={
        Path(wsi_fp).name: get_hash(wsi_fp, hash_type='sha256') for wsi_fp in log_t['WSIData']
    })
    
    # Configs
    entTestConfig = bndl.entity(f'cfgEval', other_attributes={
        'filepath': log_t['config_file'],
        'sha256': get_hash(log_t['config_file'], hash_type='sha256')
    })

    # Main Activities
    activity_name = 'predictAndEvaluateCarcinoma'
    actTestRun = bndl.activity(activity_name, other_attributes={
        'experiment_ID': log_t['eid'],
        f"git_commit_hash": log_t['git_commit_hash']
    })
    mainActivityEval.add_attributes({'hasPart': activity_name})

    # Data Entities
    predict_table_hashes = log_t['predictions']['sha256']
    predict_table_hashes.update(flatten_dict(log_t['evaluations']))
    testPredicts = bndl.entity(f"evalPredictionsWithMetricsProstate", other_attributes=predict_table_hashes)

    # Input Data Specializations
    bndl.specialization(entTrainedNetSpec, entTrainedNet)
    bndl.specialization(entTestDataset, entEvalSet)
    bndl.specialization(entWSITestData, entWSIData)
    bndl.specialization(testPredicts, entEvalSetWithMetrics)

    # Result Derivations
    bndl.wasDerivedFrom(testPredicts, entTestDataset)

    # PREDICT Activity Relationships
    bndl.used(actTestRun, entTrainedNetSpec)
    bndl.used(actTestRun, entTestDataset)
    bndl.used(actTestRun, entWSITestData)
    bndl.used(actTestRun, entTestConfig)
    bndl.wasGeneratedBy(testPredicts, actTestRun)

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