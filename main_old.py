#from tensorflow.python.client import device_lib

import numpy as np
import pandas as pd
import h5py
import json
import os
from rationai.training.models.keras_models import PretrainedNet


from rationai.utils.class_handler import get_class

from prunning.src_code.adapted_vgg import AdaptedVGG16, OnePrunableLayerAdaptedVGG16


import torch

import utils



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load_h5_store_pandas(file_path: str):
    store = pd.HDFStore(file_path)
    # data = store['train/P-2016_0057-02-0']
    # print(store.keys())
    return store

def load_h5_store_h5py(file_path: str):
    with h5py.File(file_path, 'r') as f:
        keys = f.keys()
        print(keys)
        data = f.get(keys[0])
        print(data)
    # maybe f[key_name] would work
    # what do I return?

def main_keras():

    import tensorflow as tf
    from tensorflow import keras

    from rationai.training.models.keras_models import PretrainedNet
    from rationai.utils.class_handler import get_class

    # pruning requirements
    import tensorflow_model_optimization as tfmot
    import tempfile

    #model = tf.saved_model.load("/mnt/data/home/bajger/NN_pruning/models/VGG16-TF2-DATASET-e95b-4e8f-aeea-b87904166a69/early.hdf5")
    
    
    
    
    with open('sample_experiment_test_config.json') as exp_test_config_file:
        exp_test_config = json.load(exp_test_config_file)
    
    with open('sample_experiment_train_config.json') as exp_train_config_file:
        exp_train_config = json.load(exp_train_config_file)

    model_config = PretrainedNet.Config(exp_train_config['configurations']['model'])
    model_config.parse()
    wrapped_model = PretrainedNet(model_config)
    model = wrapped_model.model
    print("MODEL PREPARED")

    print(model.layers)
    return model

    # Helper function uses `prune_low_magnitude` to make only the 
    # Dense layers train with pruning.
    def apply_pruning_to_dense(layer):
        if isinstance(layer, tf.keras.models):
            return tfmot.sparsity.keras.prune_low_magnitude(layer)
        return layer

    # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
    # to the layers of the model.
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_dense,
    )

    model_for_pruning.summary()


    exit(0)

    keras.backend.clear_session()
    
    print("SESSION CLEARED")

    # prepare references to classes and respective configs
    datagen_class = get_class(exp_train_config['definitions']['datagen'])

    # Build Datagen
    datagen_config = datagen_class.Config(exp_train_config['configurations']['datagen'])
    datagen_config.parse()
    
    generators_dict = datagen_class(datagen_config).build_from_template()
    train_generator = generators_dict['train_gen']
    valid_generator = generators_dict['valid_gen']
    train_generator.set_batch_size(128)
    valid_generator.set_batch_size(128)



    print(model.summary())



    #prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 128
    epochs = 2
    validation_split = 0.1 # 10% of training set will be used for validation set. 

    num_images = len(train_generator) * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    #model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


    model_for_pruning.summary()
    print("MODEL FOR PRUNNING COMPILED")

    logdir = tempfile.mkdtemp()
    print("Created temporary directory for logs:", logdir)

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(train_generator,# valid_generator,
                    batch_size=batch_size, epochs=epochs,# validation_split=validation_split,
                    callbacks=callbacks)
    print("TRYING TO FIT")

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)



def transplant_model_to_pytorch():
    


    model_config = PretrainedNet.Config(exp_test_config['configurations']['model'])
    model_config.parse()
    wrapped_model = PretrainedNet(model_config)
    model = wrapped_model.model

    print("KERAS MODEL PREPARED")

    print(model.layers)

    torchmodel = AdaptedVGG16()
    print("PYTORCH MODEL PREPARED?")


    corresponding_conv_layers = [
        ('block1_conv1', torchmodel.block1_conv1),
        ('block1_conv2', torchmodel.block1_conv2),
        ('block2_conv1', torchmodel.block2_conv1),
        ('block2_conv2', torchmodel.block2_conv2),
        ('block3_conv1', torchmodel.block3_conv1),
        ('block3_conv2', torchmodel.block3_conv2),
        ('block3_conv3', torchmodel.block3_conv3),
        ('block4_conv1', torchmodel.block4_conv1),
        ('block4_conv2', torchmodel.block4_conv2),
        ('block4_conv3', torchmodel.block4_conv3),
        ('block5_conv1', torchmodel.block5_conv1),
        ('block5_conv2', torchmodel.block5_conv2),
        ('block5_conv3', torchmodel.block5_conv3)
    ]

    for vgg_layer_name, new_layer in corresponding_conv_layers:
        kernels, bias = model.get_layer('vgg16').get_layer(vgg_layer_name).weights
        new_layer.weight.data = torch.tensor(kernels.numpy().transpose([3, 2, 0, 1]))
        new_layer.bias.data = torch.tensor(bias.numpy())

    weights, bias = model.get_layer('dense').weights
    torchmodel.dense.weight.data = torch.tensor(weights.numpy().transpose([1, 0]))
    torchmodel.dense.bias.data = torch.tensor(bias.numpy())

    print("PYTORCH MODEL LOADED")

    torch.save(torchmodel.state_dict(), 'transplanted-model.chkpt')


    
def load_torch_model():
    pass
   









if __name__ == '__main__':


    with open('sample_experiment_test_config.json') as exp_test_config_file:
        exp_test_config = json.load(exp_test_config_file)
    
    with open('sample_experiment_train_config.json') as exp_train_config_file:
        exp_train_config = json.load(exp_train_config_file)

    #transplant_model_to_pytorch()
    #exit()
    # prepare references to classes and respective configs
    datagen_class = get_class(exp_train_config['definitions']['datagen'])

    # Build Datagen
    datagen_config = datagen_class.Config(exp_train_config['configurations']['datagen'])
    datagen_config.parse()
    
    generators_dict = datagen_class(datagen_config).build_from_template()
    train_generator = generators_dict['train_gen']
    valid_generator = generators_dict['valid_gen']
    batch_size = 8
    train_generator.set_batch_size(batch_size)
    valid_generator.set_batch_size(batch_size)

    print(type(train_generator))
    print(len(train_generator))
    #for i in range(10):
    #     batch = train_generator[i]
    #     print(batch)
    #     print(f"({batch[0].shape}, {batch[1].shape})")

        
    
    
    

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    state_dict = torch.load('/mnt/data/home/bajger/NN_pruning/histopat/transplanted-model.chkpt')
    
    
    print(state_dict.keys())
    #print(state_dict['dense.weight'].shape)
    #print(state_dict['dense.bias'].shape)
    model = AdaptedVGG16(state_dict)
    #model.load_state_dict(state_dict=state_dict['model_state_dict'])
    model.cuda()
    #model = OnePrunableLayerAdaptedVGG16(0, state_dict)
    #print(model.dense)
    criterion = torch.nn.BCELoss()
    for i in range(len(train_generator)):
        
        input, target = train_generator[i]
        input = input.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor).view(batch_size, 1)
        ## torch async magic
        target_var = target.cuda(non_blocking =True)
        input_var = input.cuda()
        #target_var = torch.autograd.Variable(target).cuda()

        print("TRG:", target_var.shape)
        #print("INP:", input_var.size())

        # compute output
        output = model(input_var)
        print("OUTP:", output, output.size())
        #with torch.autocast('cuda'):
        loss = criterion(output, target_var)
        with torch.no_grad():
            acc = utils.binary_accuracy(output.round(), target_var.type(torch.int))
        print("LOSS:", loss, "ACC:", acc)
    train_generator.on_epoch_end()
    print("END")
    