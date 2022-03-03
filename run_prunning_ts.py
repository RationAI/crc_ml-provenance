import json
from matplotlib.pyplot import table
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

from rationai.utils.class_handler import get_class
from rationai.utils.config import build_from_config, parse_configs_recursively

from utils import binary_accuracy

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger





def load_h5_store_pandas(file_path: str):
    store = pd.HDFStore(file_path)
    return store

def showcase_ts_labels():    
    hdfs = load_h5_store_pandas('/mnt/data/home/bajger/NN_pruning/histopat/experiment_output/transfer_learning/predictions.h5')
    hdfs2 = load_h5_store_pandas('/mnt/data/home/bajger/NN_pruning/histopat/datasets/hdfs_output/hdfs_output.h5')
    i = 0
    for table_name in hdfs.keys():
        print(table_name)
        
        # metadata = hdfs2.get_storer(table_name).attrs.metadata
        
        # hdfs.get_storer(table_name).attrs.metadata = metadata
        #print(hdfs2.get_storer(table_name).attrs.metadata)
        #print(hdfs.get_storer(table_name).attrs.metadata)
        for j in range(len(hdfs[table_name]['is_cancer'])):
            values = np.asarray(hdfs[table_name]['is_cancer'][j][1:-1].split()).astype(float)
            #print(values.min(), values.max(), values.mean())
            length = values.shape[0]
            

            if length != 512:
                print("I:",i, ", J:", j, ", length:", length)
                

        i+=1
        print("Table:", i, "had", j, "values.")

        #hdfs[table_name] = hdfs[table_name].rename(columns={'is_cancer': 'is_cancer_bool', 'pred': 'is_cancer'})


def run_ts():
    with open('sample_experiment_test_config.json') as exp_test_config_file:
        exp_test_config = json.load(exp_test_config_file)
    
    with open('sample_experiment_train_config.json') as exp_train_config_file:
        exp_train_config = json.load(exp_train_config_file)

    with open('prunning_teacher_student/prunning_ts_config.json') as ts_config_file:
        ts_config = json.load(ts_config_file)



    

    # prepare references to classes and respective configs
    #train_datagen_class = get_class(exp_train_config['definitions']['datagen'])

    # Build Datagen
    # train_datagen_config = train_datagen_class.Config(exp_train_config['configurations']['datagen'])
    # train_datagen_config.parse()
    
    # train_generators_dict = train_datagen_class(train_datagen_config).build_from_template()
    batch_size = 8

    datagen_bool = build_from_config(parse_configs_recursively("datagen_bool", cfg_store=ts_config["named_configs"]))
    generators_dict_bool = datagen_bool.build_from_template()

    train_generator_bool = generators_dict_bool['train_gen']
    valid_generator_bool = generators_dict_bool['valid_gen']
    train_generator_bool.set_batch_size(batch_size)
    valid_generator_bool.set_batch_size(batch_size)
    
    datagen_ts = build_from_config(parse_configs_recursively("datagen_ts", cfg_store=ts_config["named_configs"]))
    generators_dict_ts = datagen_ts.build_from_template()

    train_generator_ts = generators_dict_ts['train_gen']
    valid_generator_ts = generators_dict_ts['valid_gen']
    train_generator_ts.set_batch_size(batch_size)
    valid_generator_ts.set_batch_size(batch_size)
    
    assert len(train_generator_ts) == len(train_generator_bool)
    assert len(valid_generator_ts) == len(valid_generator_bool)
    
    
    # test_generator.set_batch_size(batch_size)

    # print(type(test_generator))
    # print(len(test_generator))
    for i in range(3):
        xbt, ybt = train_generator_bool[i]
        xtt, ytt = train_generator_ts[i]
        print(xbt == xtt)
        xbv, ybv = valid_generator_bool[i]
        xtv, ytv = valid_generator_ts[i]
        print(xbv == xtv)
        print(f"({ybt.shape}, {ytt.shape}, {ybv.shape}, {ytv.shape})")

        
    teacher_eyes = build_from_config(parse_configs_recursively("VGG_eyes", cfg_store=ts_config["named_configs"]))
    teacher_head = build_from_config(parse_configs_recursively("VGG_head", cfg_store=ts_config["named_configs"]))
    student_model = build_from_config(parse_configs_recursively("VGG_eyes_pruned", cfg_store=ts_config["named_configs"]))
    #exit() 
    

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #state_dict = torch.load('/mnt/data/home/bajger/NN_pruning/histopat/transplanted-model.chkpt')
    
    
    #print(state_dict.keys())
    #print(state_dict['dense.weight'].shape)
    #print(state_dict['dense.bias'].shape)
    #model = AdaptedVGG16(state_dict)
    #model.load_state_dict(state_dict=state_dict['model_state_dict'])
    #model.cuda()
    #model = OnePrunableLayerAdaptedVGG16(0, state_dict)
    #print(model.dense)


    logger = TensorBoardLogger("tb_logs", name="VGG_ts")
    # trainer = Trainer(logger=logger, gpus="0,1")
    # trainer.test(teacher_model, dataloaders=test_generator)

    teacher_eyes.cuda()
    teacher_head.cuda()
    #criterion = torch.nn.CrossEntropyLoss()
    epoch_ = 0
    for i in range(len(valid_generator_ts)):
        
        input_tensor, target_tensor = valid_generator_ts[i]
        _, correct_prediction = valid_generator_bool[i]
        input_tensor = input_tensor.type(torch.FloatTensor).cuda(non_blocking=True)
        target_tensor = target_tensor.type(torch.FloatTensor).cuda()
       
        
        print("TRG:", target_tensor.shape)
        #print("INP:", input_var.size())

        # compute output
        output_tensor = teacher_eyes(input_tensor)
        #print("OUTP:", output_tensor, target_tensor)
    
        #with torch.autocast('cuda'):
        # loss = criterion(output, target_tensor)
        with torch.no_grad():
            diff = output_tensor-target_tensor
            mse_loss = F.mse_loss(output_tensor, target_tensor)
        #     #acc = binary_accuracy(output.round(), target_tensor.type(torch.int))
        # logger.experiment.add_scalar("MSE_loss", mse_loss)
        # logger.experiment.add_scalar("CE_loss", loss)

        prediction = teacher_head(output_tensor)
        loss = F.binary_cross_entropy(output_tensor, target_tensor)


        p = prediction.round()
        cp = correct_prediction.type(torch.IntTensor)
        print("PT:", p, cp)
        accuracy = binary_accuracy(p, cp)

        logger.experiment.add_scalar("LOSS", loss, i)
        logger.experiment.add_scalar("Accuracy", accuracy, i)

        print(
            "LOSS:", loss,
            "\nMSE:", mse_loss, 
            "\nDIFF:", diff,
            "\nAccuracy:", accuracy, 
            "\nPRED:", prediction)
    valid_generator_bool.on_epoch_end()
    valid_generator_ts.on_epoch_end()
    print("END")


if __name__ == "__main__":
    run_ts()
    #showcase_ts_labels()