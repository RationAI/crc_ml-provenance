#from tensorflow.python.client import device_lib

import numpy as np
import pandas as pd
import h5py
import json
import os






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





    
def main_pytorch():
    from rationai.training.models.keras_models import PretrainedNet
    from rationai.utils.class_handler import get_class

    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class AdaptedVGG16(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
            self.block1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

            self.block2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.block2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

            self.block3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.block3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
            self.block3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

            self.block4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
            self.block4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
            self.block4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

            self.block5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
            self.block5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
            self.block5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            #self.fc1 = nn.Linear(25088, 4096)
            #self.fc2 = nn.Linear(4096, 4096)
            #self.fc3 = nn.Linear(4096, 10)
            self.dropout = nn.Dropout(.5)
            self.dense = nn.Linear(512, 1)

        def forward(self, x):
            x = F.relu(self.block1_conv1(x))
            x = F.relu(self.block1_conv2(x))
            x = self.maxpool(x)

            x = F.relu(self.block2_conv1(x))
            x = F.relu(self.block2_conv2(x))
            x = self.maxpool(x)

            x = F.relu(self.block3_conv1(x))
            x = F.relu(self.block3_conv2(x))
            x = F.relu(self.block3_conv3(x))
            x = self.maxpool(x)

            x = F.relu(self.block4_conv1(x))
            x = F.relu(self.block4_conv2(x))
            x = F.relu(self.block4_conv3(x))
            x = self.maxpool(x)

            x = F.relu(self.block5_conv1(x))
            x = F.relu(self.block5_conv2(x))
            x = F.relu(self.block5_conv3(x))
            x = self.maxpool(x)

            
            x = F.max_pool2d(x, kernel_size=x.size()[2:])  # Global max pooling
            
            x = self.dropout(x)
            x = F.sigmoid(self.dense(x))
        
            return x


    model_config = PretrainedNet.Config(exp_train_config['configurations']['model'])
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

    #torch.save({'model_state_dict': torchmodel.state_dict()}, 'transplanted-model.chkpt')

    exit(0)

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




if __name__ == '__main__':
    with open('sample_experiment_test_config.json') as exp_test_config_file:
        exp_test_config = json.load(exp_test_config_file)
    
    with open('sample_experiment_train_config.json') as exp_train_config_file:
        exp_train_config = json.load(exp_train_config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #load_torch_model()
    main_pytorch()