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
from rationai.utils.provenance import flatten_lists


def export_provenance(log_fp: Path) -> None:
    doc = prepare_document()
    log_t = parse_log(log_fp)
    USE_VALIDATION = log_t['config']['valid_generator'] is not None

    # Creating training bundle
    bndl = doc.bundle(f"{NAMESPACE_TRAINING}:bundle_training")

    ###                                                                    ###
    #                     Creating Backbone Part                             #
    ##                                                                     ###

    # Receiver connector
    conn_train_set = bndl.entity(f"{NAMESPACE_PREPROC}:datasetTrainConnector", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverConnector",
        f"{NAMESPACE_COMMON_MODEL}:senderBundleId": f"{NAMESPACE_PREPROC}:bundle_preproc",
        f"{NAMESPACE_COMMON_MODEL}:senderServiceUri": f"#URI#"
    })

    # Sender connector
    conn_trained_model_connector = bndl.entity(f'{NAMESPACE_TRAINING}:trainedModelConnector', other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderConnector",
        f"{NAMESPACE_COMMON_MODEL}:receiverBundleId": f"{NAMESPACE_EVAL}:bundle_eval",
        f"{NAMESPACE_COMMON_MODEL}:receiverServiceUri": f"#URI#"
    })

    # Receiver Agent
    recAgent = bndl.agent(f"{NAMESPACE_PREPROC}:receiverAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiverAgent"
    })

    # Sender Agent
    sendAgent = bndl.agent(f"{NAMESPACE_TRAINING}:senderAgent", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:senderAgent"
    })

    # Receipt Activity
    act_receipt = bndl.activity(f"{NAMESPACE_TRAINING}:receipt", other_attributes={
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:receiptActivity",
    })

    # Main Activity - they are ordered (Splitter -> Generators -> Train Iters -> Valid Iters)
    act_epochs = len(log_t['iters'].keys()) - 1  # Note: Generators' on_epoch_end() will create one additional epochs in the logs.

    other_attributes = {
        f"{NAMESPACE_PROV}:type": f"{NAMESPACE_COMMON_MODEL}:mainActivity",
        'git_commit_hash': log_t['git_commit_hash']
    }

    has_part = 0
    if USE_VALIDATION:
        other_attributes[f'{NAMESPACE_DCT}:hasPart{has_part}'] = f'{NAMESPACE_TRAINING}:datasetSplitting'
        has_part += 1

    other_attributes[f'{NAMESPACE_DCT}:hasPart{has_part}'] = f'{NAMESPACE_TRAINING}:trainGenerator'
    has_part += 1

    if USE_VALIDATION:
         other_attributes[f'{NAMESPACE_DCT}:hasPart{has_part}'] = f'{NAMESPACE_TRAINING}:validGenerator'
         has_part += 1

    for iter_i in range(act_epochs):
        other_attributes[f'{NAMESPACE_DCT}:hasPart{has_part}'] = f'{NAMESPACE_TRAINING}:trainIter{iter_i}'
        has_part += 1

    if USE_VALIDATION:
        for iter_i in range(act_epochs):
            other_attributes[f'{NAMESPACE_DCT}:hasPart{has_part}'] = f'{NAMESPACE_TRAINING}:validIter{iter_i}'
            has_part += 1


    act_training = bndl.activity(f'{NAMESPACE_TRAINING}:training', other_attributes=other_attributes)

    # Data Entity Node
    ent_train_group = bndl.entity(f'{NAMESPACE_TRAINING}:dataset')

    # Establish relationships between backbones nodes
    bndl.used(act_receipt, conn_train_set)
    bndl.wasGeneratedBy(ent_train_group, act_receipt)
    bndl.used(act_training, ent_train_group)
    bndl.wasGeneratedBy(conn_trained_model_connector, act_training)
    bndl.attribution(conn_train_set, sendAgent)
    bndl.attribution(conn_trained_model_connector, recAgent)


    ###                                                                    ###
    #                Creating Domain-specific Part                           #
    ###                                                                    ###

    # Specialized Dataset
    ent_train_group_prostate = bndl.entity(f'{NAMESPACE_TRAINING}:trainData' ,other_attributes={
        'file': log_t['config']['configurations']['datagen']['data_sources']['_data'],
        'groups': ', '.join(log_t['config']['configurations']['datagen']['data_sources']['definitions']['train_ds']['keys']),
        'sha256': get_sha256(log_t['config']['configurations']['datagen']['data_sources']['_data'])
    })
    bndl.specializationOf(ent_train_group_prostate, ent_train_group)

    # Required Training Slides
    ent_train_slides = bndl.entity(f'{NAMESPACE_TRAINING}:trainSlides', other_attributes=log_t['splits']['train_gen'])

    # Required Activity Node
    train_dg_cfg = log_t['config']['configurations']['datagen']['generators']['train_gen']['components']
    train_dg_cfg = flatten_lists(train_dg_cfg)
    act_train_sampler = bndl.activity(f'{NAMESPACE_TRAINING}:trainGenerator', other_attributes=train_dg_cfg)

    if USE_VALIDATION:
        # Optional Splitting Activity Node
        split_cfg = log_t['config']['configurations']['datagen']['data_sources']['definitions']['train_ds']
        split_cfg = flatten_lists(split_cfg)
        act_dataset_splitting = bndl.activity(f'{NAMESPACE_TRAINING}:datasetSplitting', other_attributes=split_cfg)
        bndl.used(act_dataset_splitting, ent_train_group_prostate)
        bndl.wasGeneratedBy(ent_train_slides, act_dataset_splitting)

        # Optional Validation Slides
        ent_valid_slides = bndl.entity(f'{NAMESPACE_TRAINING}:validSlides', other_attributes=log_t['splits']['valid_gen'])

        # Optional Activity Node
        valid_dg_cfg = log_t['config']['configurations']['datagen']['generators']['valid_gen']['components']
        valid_dg_cfg = flatten_lists(valid_dg_cfg)
        act_valid_sampler = bndl.activity(f'{NAMESPACE_TRAINING}:validGenerator', other_attributes=valid_dg_cfg)

        bndl.wasGeneratedBy(ent_valid_slides, act_dataset_splitting)
        bndl.wasDerivedFrom(ent_valid_slides, ent_train_group_prostate)
        bndl.used(act_valid_sampler, ent_valid_slides)

    bndl.used(act_train_sampler, ent_train_slides)
    bndl.wasDerivedFrom(ent_train_slides, ent_train_group_prostate)

    # Initial Model
    init_model = bndl.entity(f'{NAMESPACE_TRAINING}:modelInit', other_attributes={
        'model': log_t['config']['definitions']['model']
    })

    # Init Checkpoint
    init_checkpoint = bndl.entity(f'{NAMESPACE_TRAINING}:checkpointInit', other_attributes={
        'filepath': log_t['init_checkpoint_file']
    })

    bndl.wasGeneratedBy(init_checkpoint, init_model)
    bndl.specializationOf(init_checkpoint, conn_trained_model_connector)

    last_model = init_model

    for epoch in range(act_epochs):
        train_tiles_subset = bndl.entity(f'{NAMESPACE_TRAINING}:trainTilesSubset{epoch}', other_attributes={
            "sampled_epoch_sha256": f"{log_t['iters'][f'{epoch}']['train_gen']['sha256']}"
        })
        bndl.wasGeneratedBy(train_tiles_subset, act_train_sampler)
        bndl.wasDerivedFrom(train_tiles_subset, ent_train_slides)

        act_train_iter = bndl.activity(f'{NAMESPACE_TRAINING}:trainIter{epoch}', other_attributes=log_t['iters'][f'{epoch}']['metrics']['train'])
        result_model = bndl.entity(f'{NAMESPACE_TRAINING}:modelIter{epoch+1}')

        # First iteration
        bndl.used(act_train_iter, train_tiles_subset)
        bndl.used(act_train_iter, last_model)
        bndl.wasGeneratedBy(result_model, act_train_iter)
        bndl.wasDerivedFrom(result_model, last_model)

        if USE_VALIDATION:
            valid_tiles_subset = bndl.entity(f'{NAMESPACE_TRAINING}:validTilesSubset{epoch}', other_attributes={
                "sampled_epoch_sha256": f"{log_t['iters'][f'{epoch}']['valid_gen']['sha256']}"
            })
            bndl.wasGeneratedBy(valid_tiles_subset, act_valid_sampler)
            bndl.wasDerivedFrom(valid_tiles_subset, ent_valid_slides)

            act_valid_iter1 = bndl.activity(f'{NAMESPACE_TRAINING}:validIter{epoch}', other_attributes=log_t['iters'][f'{epoch}']['metrics']['valid'])

            bndl.used(act_valid_iter1, valid_tiles_subset)
            bndl.used(act_valid_iter1, result_model)

            for ckpt in log_t['iters'][f'{epoch}']['checkpoints']:
                if bool(log_t['iters'][f'{epoch}']['checkpoints'][ckpt]['valid']):
                    checkpoint = bndl.entity(f"{NAMESPACE_TRAINING}:ckpt_{Path(log_t['iters'][f'{epoch}']['checkpoints'][ckpt]['filepath']).stem}_{epoch}", other_attributes={
                        'filepath': log_t['iters'][f'{epoch}']['checkpoints'][ckpt]['filepath']
                    })
                    bndl.wasGeneratedBy(checkpoint, act_valid_iter1)
                    bndl.specializationOf(checkpoint, conn_trained_model_connector)


        last_model = result_model

    export_to_image(bndl, 'training')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--log_fp', type=Path, required=True, help='Path to provenanace log')
    args = parser.parse_args()

    export_provenance(args.log_fp)

    