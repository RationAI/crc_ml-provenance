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
from rationai.utils.provenance import flatten_lists


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
    
    log_fp =  (experiment_dir / BUNDLE_TRAIN).with_suffix('.log')
    assert log_fp.exists(), f'Execution log not found: {log_fp}'
    
    doc = prov.ProvDocument()
    add_namespaces(doc)
    log_t = parse_log(log_fp)
    USE_VALIDATION = log_t['config']['valid_generator'] is not None

    # Creating training bundle
    bndl = doc.bundle(f"{PROVN_NAMESPACE}:{BUNDLE_TRAIN}")
    add_namespaces(bndl)

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###

    # Backward Connectors
    entity_identifier = 'trainIndexROITablesConnector'
    connEntTrainSet = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:backwardConnector"),
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": bndl.valid_qualified_name(f"{PROVN_NAMESPACE}:{BUNDLE_PREPROC}"),
        #f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": bndl.valid_qualified_name(f'{DEFAULT_NAMESPACE_URI}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}'),
    })
    
    entity_identifier = 'WSIDataConnectorTrain'
    connEntWSIData = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:backwardConnector"),
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": bndl.valid_qualified_name(f"{BACKWARD_PROVN_NAMESPACE}:{BACKWARD_BUNDLE}"),
        #f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": bndl.valid_qualified_name(f'{BACKWARD_PROVN_NAMESPACE}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{BACKWARD_PROVN_NAMESPACE}:{BUNDLE_META}'),
    })

    # Forward Connectors
    entity_identifier = 'trainedModelConnector'
    connEntTrainedModel = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:forwardConnector"),
        f"{NAMESPACE_COMMON_MODEL}:receiverBundleId": bndl.valid_qualified_name(f"{PROVN_NAMESPACE}:{BUNDLE_EVAL}"),
        #f"{NAMESPACE_COMMON_MODEL}:receiverServiceUri": f'{DEFAULT_NAMESPACE_URI}',
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
    
    # Current Connectors
    entity_identifier = 'trainIndexROITables'
    entTrainSet = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f'{NAMESPACE_COMMON_MODEL}:currentConnector'),
        f"{NAMESPACE_COMMON_MODEL}:currentBundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_TRAIN}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
    
    entity_identifier = 'WSIDataTrain'
    entWSIData = bndl.entity(bndl.valid_qualified_name(f"{DOI_NAMESPACE}:{entity_identifier}"), other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f'{NAMESPACE_COMMON_MODEL}:currentConnector'),
        f"{NAMESPACE_COMMON_MODEL}:currentBundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_TRAIN}'),
        f"{NAMESPACE_COMMON_MODEL}:metabundle": bndl.valid_qualified_name(f'{PROVN_NAMESPACE}:{BUNDLE_META}')
    })
        

    # Agents
    agentRDC = bndl.agent("researchDataCentre", other_attributes={
        f"{NAMESPACE_PROV}:type":  bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:senderAgent"),
        f"{NAMESPACE_PROV}:type":  bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:receiverAgent")
    })
    
    agentMMCI = bndl.agent("pathologyDepartment", other_attributes={
        f"{NAMESPACE_PROV}:type":  bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:senderAgent")
    })

    # Receipt Activity
    actReceiptTrainDataset = bndl.activity(f"receiptROITables", other_attributes={
        f"{NAMESPACE_PROV}:type":  bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:receiptActivity"),
    })
    
    actReceiptWSIData = bndl.activity(f"receiptWSIDataset", other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name( f"{NAMESPACE_COMMON_MODEL}:receiptActivity"),
    })

    # Main Activity
    # Note: Generators' on_epoch_end() will create one additional epoch record in the logs at the end of last epoch.
    act_epochs = len(log_t['iters'].keys()) - 1  

    act_training = bndl.activity(f'training', other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:mainActivity"),
        'git_commit_hash': log_t['git_commit_hash']
    })

    if USE_VALIDATION:
        act_training.add_attributes({f'{NAMESPACE_DCT}:hasPart': f'datasetSplitting'})

    act_training.add_attributes({f'{NAMESPACE_DCT}:hasPart': f'trainGenerator'})

    if USE_VALIDATION:
        act_training.add_attributes({f'{NAMESPACE_DCT}:hasPart': f'validGenerator'})

    for iter_i in range(act_epochs):
        act_training.add_attributes({f'{NAMESPACE_DCT}:hasPart': f'trainIter{iter_i}'})

    if USE_VALIDATION:
        for iter_i in range(act_epochs):
            act_training.add_attributes({f'{NAMESPACE_DCT}:hasPart': f'validIter{iter_i}'})
    
    # Agent Relations
    bndl.wasAttributedTo(connEntTrainSet, agentRDC)
    bndl.wasAttributedTo(connEntTrainedModel, agentRDC)
    bndl.wasAttributedTo(connEntWSIData, agentMMCI)
    
    # Receipt Relations
    bndl.used(actReceiptTrainDataset, connEntTrainSet)
    bndl.used(actReceiptWSIData, connEntWSIData)
    
    bndl.wasInvalidatedBy(connEntTrainSet, actReceiptTrainDataset)
    bndl.wasInvalidatedBy(connEntWSIData, actReceiptWSIData)
    
    bndl.wasGeneratedBy(entTrainSet, actReceiptTrainDataset)
    bndl.wasGeneratedBy(entWSIData, actReceiptWSIData)
    
    bndl.wasDerivedFrom(entTrainSet, connEntTrainSet)
    bndl.wasDerivedFrom(entWSIData, connEntWSIData)
    
    # Main Activity Relations
    bndl.used(act_training, entTrainSet)
    bndl.used(act_training, entWSIData)
    
    bndl.wasGeneratedBy(connEntTrainedModel, act_training)
    
    bndl.wasDerivedFrom(connEntTrainedModel, entTrainSet)
    bndl.wasDerivedFrom(connEntTrainedModel, entWSIData)


    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    # Specialized Dataset
    entTrainSetProstate = bndl.entity('trainIndexROITableProstate', other_attributes={
        'file': log_t['config']['configurations']['datagen']['data_sources']['_data'],
        'groups': ', '.join(log_t['config']['configurations']['datagen']['data_sources']['definitions']['train_ds']['keys']),
        'sha256': get_hash(log_t['config']['configurations']['datagen']['data_sources']['_data'], hash_type='sha256')
    } | log_t['splits']['train_gen'])
    bndl.specializationOf(entTrainSetProstate, entTrainSet)
    
    entWSIDataProstate = bndl.entity('WSIDataTrainProstate', other_attributes={
        Path(wsi_fp).name: get_hash(wsi_fp, hash_type='sha256') for wsi_fp in log_t['WSIData']
    })
    bndl.specializationOf(entWSIDataProstate, entWSIData)

    # Required Activity Node
    train_dg_cfg = log_t['config']['configurations']['datagen']['generators']['train_gen']['components']
    train_dg_cfg = flatten_lists(train_dg_cfg)
    act_train_sampler = bndl.activity(f'trainGenerator', other_attributes=train_dg_cfg)

    if USE_VALIDATION:
        # Optional Splitting Activity Node
        split_cfg = log_t['config']['configurations']['datagen']['data_sources']['definitions']['train_ds']
        split_cfg = flatten_lists(split_cfg)
        act_dataset_splitting = bndl.activity(f'datasetSplitting', other_attributes=split_cfg)
        bndl.used(act_dataset_splitting, entTrainSetProstate)
        bndl.wasGeneratedBy(entWSIDataProstate, act_dataset_splitting)

        # Optional Validation Slides
        ent_valid_slides = bndl.entity(f'validSlides', other_attributes=log_t['splits']['valid_gen'])

        # Optional Activity Node
        valid_dg_cfg = log_t['config']['configurations']['datagen']['generators']['valid_gen']['components']
        valid_dg_cfg = flatten_lists(valid_dg_cfg)
        act_valid_sampler = bndl.activity(f'validGenerator', other_attributes=valid_dg_cfg)

        bndl.wasGeneratedBy(ent_valid_slides, act_dataset_splitting)
        bndl.wasDerivedFrom(ent_valid_slides, entTrainSetProstate)
        bndl.used(act_valid_sampler, ent_valid_slides)

    bndl.used(act_train_sampler, entWSIDataProstate)
    bndl.used(act_train_sampler, entTrainSetProstate)

    # Trained Model Collection Entity
    ent_trained_model_collection = bndl.entity(f'trainedModelCollection', other_attributes={
        f"{NAMESPACE_PROV}:type": bndl.valid_qualified_name(f"{NAMESPACE_COMMON_MODEL}:Collection")
    })
    
    bndl.specializationOf(ent_trained_model_collection, connEntTrainedModel)
    
    # Initial Model
    init_model = bndl.entity(f'modelInit', other_attributes={
        'model': log_t['config']['definitions']['model'],
        'filepath': log_t['init_checkpoint_file']
    })
    
    bndl.hadMember(ent_trained_model_collection, init_model)

    last_model = init_model

    for epoch in range(act_epochs):
        MODEL_IS_MEMBER_FLAG = False
        train_tiles_subset = bndl.entity(f'trainTilesSubset{epoch}', other_attributes={
            "sampled_epoch_sha256": f"{log_t['iters'][f'{epoch}']['train_gen']['sha256']}"
        })
        bndl.wasGeneratedBy(train_tiles_subset, act_train_sampler)
        bndl.wasDerivedFrom(train_tiles_subset, entWSIDataProstate)
        bndl.wasDerivedFrom(train_tiles_subset, entTrainSetProstate)

        act_train_iter = bndl.activity(f'trainIter{epoch}', other_attributes=log_t['iters'][f'{epoch}']['metrics']['train'])
        result_model = bndl.entity(f'modelIter{epoch+1}')

        # First iteration
        bndl.used(act_train_iter, train_tiles_subset)
        bndl.used(act_train_iter, last_model)
        bndl.wasGeneratedBy(result_model, act_train_iter)
        bndl.wasDerivedFrom(result_model, last_model)

        if USE_VALIDATION:
            valid_tiles_subset = bndl.entity(f'validTilesSubset{epoch}', other_attributes={
                "sampled_epoch_sha256": f"{log_t['iters'][f'{epoch}']['valid_gen']['sha256']}"
            })
            bndl.wasGeneratedBy(valid_tiles_subset, act_valid_sampler)
            bndl.wasDerivedFrom(valid_tiles_subset, ent_valid_slides)

            act_valid_iter1 = bndl.activity(f'validIter{epoch}', other_attributes=log_t['iters'][f'{epoch}']['metrics']['valid'])

            bndl.used(act_valid_iter1, valid_tiles_subset)
            bndl.used(act_valid_iter1, result_model)

        for ckpt in log_t['iters'][f'{epoch}']['checkpoints']:
            if bool(log_t['iters'][f'{epoch}']['checkpoints'][ckpt]['valid']):
                result_model = bndl.entity(f'modelIter{epoch+1}', other_attributes={
                    Path(log_t['iters'][f'{epoch}']['checkpoints'][ckpt]['filepath']).stem: log_t['iters'][f'{epoch}']['checkpoints'][ckpt]['filepath']
                })
                
                if not MODEL_IS_MEMBER_FLAG:
                    bndl.hadMember(ent_trained_model_collection, result_model)
                    MODEL_IS_MEMBER_FLAG = True


        last_model = result_model
    
    subdirs = ['', 'graph', 'json', 'provn']
    for subdir in subdirs:
        if not (OUTPUT_DIR / subdir).exists():
            (OUTPUT_DIR / subdir).mkdir(parents=True)

    export_to_image(bndl, (OUTPUT_DIR / 'graph' / BUNDLE_TRAIN).with_suffix('.png'))
    export_to_file(doc, (OUTPUT_DIR / 'json' / BUNDLE_TRAIN).with_suffix('.json'), format='json')
    export_to_file(doc, OUTPUT_DIR / 'provn' / BUNDLE_TRAIN, format='provn')
    
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
            'local_png': str((OUTPUT_DIR / 'graph' / BUNDLE_TRAIN).with_suffix('.png')),
            'remote_png': str(GRAPH_NAMESPACE_URI + str(Path(BUNDLE_TRAIN).with_suffix('.png'))),
            'local_provn': str(OUTPUT_DIR / 'provn' / BUNDLE_TRAIN),
            'remote_provn': str(PROVN_NAMESPACE_URI + BUNDLE_TRAIN)
        }
    }
    with open(experiment_dir / f'{BUNDLE_TRAIN}.log', 'w') as json_out:
        json.dump(output_log, json_out, indent=3)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Path to provenanace log of a train run')
    parser.add_argument('--eid', type=str, required=True, help='Execution UUID')
    args = parser.parse_args()    
    
    export_provenance(args.config_fp)

    