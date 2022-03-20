import json
from typing import NoReturn, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from rationai.datagens.generators import BaseGeneratorPytorch

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
        print(hdfs2.get_storer(table_name).attrs.metadata)
        #hdfs.get_storer(table_name).attrs.metadata['is_cancer_bool'] = hdfs.get_storer(table_name).attrs.metadata['is_cancer']
        print(hdfs.get_storer(table_name).attrs.metadata)
        # for j in range(len(hdfs[table_name]['is_cancer'])):
        #     values = np.asarray(hdfs[table_name]['is_cancer'][j][1:-1].split()).astype(float)
        #     #print(values.min(), values.max(), values.mean())
        #     length = values.shape[0]
            

        #     if length != 512:
        #         print("I:",i, ", J:", j, ", length:", length)
                

        i+=1
        #print("Table:", i, "had", j, "values.")

        #hdfs[table_name] = hdfs[table_name].rename(columns={'is_cancer': 'is_cancer_bool', 'pred': 'is_cancer'})

class GeneratorAdHocWrapper(Dataset):
    """Generates inputs from first generator and output targets from all generators.
    Assumes generators generate data with aligned indices

    Args:
        Dataset (_type_): PyTorch Dataset class for generating ztraining data
    """
    def __init__(self, *generators: BaseGeneratorPytorch) -> None:
        super().__init__()
        self.generators = generators
    
    def set_batch_size(self, batch_size: int):
        """
        TODO: Missing docstring.
        """
        for g_ in self.generators:
            g_.batch_size = batch_size

    def __len__(self) -> int:
        size = self.generators[0].__len__()
        for g_ in self.generators:
            assert g_.__len__() == size
        return size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """Get data batch at `index` from `self.epoch_samples`.

        Parameters
        ----------
        index : int
            The position of the batch.

        Return
        ------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple representing a batch with the format (input_data, label_data).
        """
        #if self.batch_size > 1:
        xs, ys = self.generators[0].__gettem__(index)
        all_ys = [ys]
        for g_ in self.generators:
            xs_, ys_ = g_.__gettem__(index)
            assert xs.eq(xs_)
            all_ys.append(ys_)
        return (xs, *all_ys)


    def on_epoch_end(self) -> NoReturn:
        """
        TODO: Missing docstring.
        """
        for g_ in self.generators:
            g_.on_epoch_end()




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
    batch_size = 1

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

    #valid_generator_both = GeneratorAdHocWrapper(valid_generator_ts, valid_generator_bool)
    #train_generator_both = GeneratorAdHocWrapper(train_generator_ts, train_generator_bool)
    
    #print(len(train_generator_ts), len(train_generator_bool), len(valid_generator_ts), len(valid_generator_bool))

    #assert len(train_generator_ts) == len(train_generator_bool)
    #assert len(valid_generator_ts) == len(valid_generator_bool)
    
    
    # test_generator.set_batch_size(batch_size)

    # print(type(test_generator))
    # print(len(test_generator))
    # for i in range(3):
    #     xbt, ybt = train_generator_bool[i]
    #     xtt, ytt = train_generator_ts[i]
    #     print(xbt == xtt)
    #     xbv, ybv = valid_generator_bool[i]
    #     xtv, ytv = valid_generator_ts[i]
    #     print(xbv == xtv)
    #     print(f"({ybt.shape}, {ytt.shape}, {ybv.shape}, {ytv.shape})")

    pruned_model_name = "VGG_eyes_pruned"
    #teacher_eyes = build_from_config(parse_configs_recursively("VGG_eyes", cfg_store=ts_config["named_configs"]))
    teacher_head = build_from_config(parse_configs_recursively("VGG_head", cfg_store=ts_config["named_configs"]))
    student_eyes = build_from_config(parse_configs_recursively(pruned_model_name, cfg_store=ts_config["named_configs"]))
    
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


    logger = TensorBoardLogger("tb_logs", name=pruned_model_name)
    # trainer = Trainer(logger=logger, gpus="0,1")
    # trainer.train(teacher_model, dataloaders=train_generator_both)

    #teacher_eyes.cuda()
    #teacher_head.cuda()
    student_eyes.cuda()
    optimizer = torch.optim.Adam(student_eyes.parameters(), lr=1e-3, betas=(0.9, 0.999))
    optimizer.zero_grad()
    #criterion = torch.nn.CrossEntropyLoss()
    
    train_epoch_sample_count = len(train_generator_ts)
    valid_epoch_sample_count = len(valid_generator_ts)
    
    val_not_decreases = True
    best_loss = None
    best_accuracy = None
    epoch_ = 0
    while val_not_decreases:
        average_loss = 0.0
        average_accuracy = 0.0
        student_eyes.train()
        for i in range(train_epoch_sample_count):
            # zero the parameter gradients
           

            # get the data batch
            input_tensor, target_tensor = train_generator_ts[i]
            _, correct_prediction = train_generator_bool[i]
            input_tensor = input_tensor.type(torch.FloatTensor).cuda(non_blocking=True)
            target_tensor = target_tensor.type(torch.FloatTensor).cuda()
            
            
            #print("TRG:", target_tensor.shape)
            #print("INP:", input_var.size())

            # compute output
            output_tensor = student_eyes(input_tensor)
            #print("OUTP:", output_tensor, target_tensor)
        
            #with torch.autocast('cuda'):
            # loss = criterion(output, target_tensor)
            
            #diff = output_tensor-target_tensor
            mse_loss = F.mse_loss(output_tensor, target_tensor)
            output_tensor = output_tensor.cpu()
            prediction = teacher_head(output_tensor)
            #loss = F.binary_cross_entropy_with_logits(output_tensor, target_tensor)


            p = prediction.view(-1).round().type(torch.IntTensor)
            cp = correct_prediction.type(torch.IntTensor)
            #print("PT:", p, p.size(), cp, cp.size())
            accuracy = binary_accuracy(p, cp)

            

            # print(
            #     #"LOSS:", loss,
            #     "\nMSE:", mse_loss, 
            #     #"\nDIFF:", diff,
            #     "\nAccuracy:", accuracy, 
            #     "\nPRED:", prediction)
            mse_loss.backward()
            
            if i % 32 == 0:
                optimizer.step()
                optimizer.zero_grad()

            #logger.experiment.add_scalar("LOSS", loss, i)
            logger.experiment.add_scalar("Train example MSE", mse_loss, i)
            logger.experiment.add_scalar("Train example Accuracy", accuracy, i)
            

        train_generator_bool.on_epoch_end()
        train_generator_ts.on_epoch_end()
        
        student_eyes.eval()
        with torch.no_grad():
            for i in range(valid_epoch_sample_count):
                input_tensor, target_tensor = valid_generator_ts[i]
                _, correct_prediction = valid_generator_bool[i]
                input_tensor = input_tensor.type(torch.FloatTensor).cuda(non_blocking=True)
                target_tensor = target_tensor.type(torch.FloatTensor).cuda()

                output_tensor = student_eyes(input_tensor)
                mse_loss = F.mse_loss(output_tensor, target_tensor)

                output_tensor = output_tensor.cpu()
                prediction = teacher_head(output_tensor)

                p = prediction.view(-1).round().type(torch.IntTensor)
                cp = correct_prediction.type(torch.IntTensor)
                accuracy = binary_accuracy(p, cp)

                average_accuracy += accuracy
                average_loss += mse_loss

                
                logger.experiment.add_scalar("Validation example MSE", mse_loss, i)
                logger.experiment.add_scalar("Validation example Accuracy", accuracy, i)

        valid_generator_bool.on_epoch_end()
        valid_generator_ts.on_epoch_end()

        average_accuracy /= valid_epoch_sample_count
        average_loss /= valid_epoch_sample_count
    
        logger.experiment.add_scalar("AVG Validation MSE - epoch", average_loss, epoch_)
        logger.experiment.add_scalar("AVG Validation Accuracy - epoch", average_accuracy, epoch_)

        if best_loss is None or average_loss < best_loss:
            best_loss = average_loss
        if best_accuracy is None or average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            torch.save(student_eyes.state_dict(), f'/mnt/data/home/bajger/NN_pruning/histopat/prunning_teacher_student/results/{pruned_model_name}.chckpt')
        

        print("END epoch", epoch_, "AVG Accuracy", average_accuracy, "Avg loss", average_loss)
        epoch_ += 1


if __name__ == "__main__":
    run_ts()
    #showcase_ts_labels()