import torch


import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


import logging

class LitVGG16(pl.LightningModule):
    def __init__(self, source_state_dict_path: str=None) -> None:
        super().__init__()
        self.eyes = VGG16_eyes(source_state_dict_path)
        self.head = VGG16_head(source_state_dict_path)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y_eyes, ys_bool = batch
        x = self.eyes(x)
        mse_loss = F.mse_loss(x, y_eyes)
        x = self.head(x)
        bce_loss = F.binary_cross_entropy_with_logits(x, ys_bool)
        

        return {
            "loss": mse_loss,
            "bce_loss": bce_loss,
            "mse_loss": mse_loss,
            "total": x.size(0)
        }
    
    def training_epoch_end(self, outputs):
        # the function is called after every epoch is completed
        # calculating average loss 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_bce_loss = torch.stack([x['bce_loss'] for x in outputs]).mean()
        
        # calculating correect and total predictions
        
        total=sum([x["total"] for  x in outputs])
         # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",avg_loss,self.current_epoch)
        self.logger.experiment.add_scalar("BCELoss/Train",avg_bce_loss,self.current_epoch)


        #self.logger.experiment.add_scalar("Accuracy/Train",correct/total,self.current_epoch)

        epoch_dictionary={'loss': avg_loss}

        return epoch_dictionary




class VGG16_eyes(nn.Module):
    def __init__(self, source_state_dict_path: str=None) -> None:
        super().__init__()
        
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)  
        self.block1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  

        self.block2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  
        self.block2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.block3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.block3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.block3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.block4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.block4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.block4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.block5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.block5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.block5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # --> 512 * W * H

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # reused max_pooling module


        #self.eyes.add_module("maxpool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  # reused max_pooling module

        if source_state_dict_path is not None:
            print("Loading weights from a supplied state_dict...")
            source_state_dict = torch.load(source_state_dict_path)
           # get a deep copy of the existing parameters
            new_weights = self.state_dict()
            
            _missing_keys = []
            _loaded_params = []
            _unexpected_keys = set(source_state_dict.keys())
            for params_name in new_weights.keys():
                if params_name in source_state_dict:
                    new_weights[params_name] = source_state_dict[params_name]
                    _loaded_params.append(params_name)
                    _unexpected_keys.remove(params_name)
                else:
                    _missing_keys.append(params_name)
            #load the updated weights
            self.load_state_dict(new_weights)
            print("Weights loaded:", _loaded_params)
            print("Weights not expected:", _unexpected_keys)
            print("Weights not present:", _missing_keys)

        
    def forward(self, x: torch.Tensor):
        # input 3 * W * H 
        x = F.relu(self.block1_conv1(x))  # --> 64 * W * H
        x = F.relu(self.block1_conv2(x))
        x = self.maxpool(x)               # --> 64 * W/2 * H/2

        x = F.relu(self.block2_conv1(x))
        x = F.relu(self.block2_conv2(x))
        x = self.maxpool(x)               # --> 128 * W/4 * H/4

        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_conv3(x))
        x = self.maxpool(x)               # --> 256 * W/8 * H/8

        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_conv3(x))
        x = self.maxpool(x)               # --> 512 * W/16 * H/16

        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))
        #x = self.maxpool(x)               # --> 512 * W/32 * H/32
        
       
        # Global max pooling --> from each filter response take the maximum value (positive signal)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])  # results in a vector of size 512 * 1 * 1
        x = x.view(x.size(0), -1)
        
        return x

    
        


class SmallerVGG_eyes(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        _relu_inplace=True
        _relu = nn.ReLU(inplace=_relu_inplace)
        _maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.eyes = nn.Sequential()
        #input batch * 3 * W * H
        self.eyes.add_module("block1_conv1", nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=2))
        self.eyes.add_module('block1_relu1', _relu)  # reused ReLU module
        self.eyes.add_module("block1_maxpool", _maxpool)

        self.eyes.add_module("block2_conv1", nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2))
        self.eyes.add_module('block2_relu1', _relu)  # reused ReLU module
        self.eyes.add_module("block2_maxpool", _maxpool)

        self.eyes.add_module("block3_conv1", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=3))
        self.eyes.add_module('block3_relu1', _relu)  # reused ReLU module
        self.eyes.add_module("block3_maxpool", _maxpool)

        self.eyes.add_module("block4_conv1", nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2))
        self.eyes.add_module('block4_relu1', _relu)  # reused ReLU module
        self.eyes.add_module("block4_maxpool", _maxpool)

    def forward(self, x):
        x = self.eyes(x)
         # Global max pooling --> from each filter response take the maximum value (positive signal)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])  # results in a vector of size 512 * 1 * 1
        x = x.view(x.size(0), -1)
        return x


class VGG16_head(nn.Module):
    def __init__(self, source_state_dict_path: str=None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(.5)
        self.dense = nn.Linear(512, 1)

        if source_state_dict_path is not None:
            print("Loading weights from a supplied state_dict...")
            source_state_dict = torch.load(source_state_dict_path)
           # get a deep copy of the existing parameters
            new_weights = self.state_dict()
            
            _missing_keys = []
            _loaded_params = []
            _unexpected_keys = set(source_state_dict.keys())
            for params_name in new_weights.keys():
                if params_name in source_state_dict:
                    new_weights[params_name] = source_state_dict[params_name]
                    _loaded_params.append(params_name)
                    _unexpected_keys.remove(params_name)
                else:
                    _missing_keys.append(params_name)
            #load the updated weights
            self.load_state_dict(new_weights)
            print("Weights loaded:", _loaded_params)
            print("Weights not expected:", _unexpected_keys)
            print("Weights not present:", _missing_keys)
        
    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.sigmoid(x)
        return x

    
        
