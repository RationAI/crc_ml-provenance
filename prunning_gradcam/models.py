from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class GMaxPool2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Tensor):
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.squeeze()
        return x


class SequentialVGG16(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        _relu_inplace=True
        _relu = nn.ReLU(inplace=_relu_inplace)
        _maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.add_module('block1_conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.add_module('block1_relu1', _relu)  # reused ReLU module
        self.add_module('block1_conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.add_module('block1_relu2', _relu)  # reused ReLU module
        self.add_module('block1_maxpool', _maxpool)  # reused max_pooling module

        self.add_module('block2_conv1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))  
        self.add_module('block2_relu1', _relu)  # reused ReLU module
        self.add_module('block2_conv2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.add_module('block2_relu2', _relu)  # reused ReLU module
        self.add_module('block2_maxpool', _maxpool)  # reused max_pooling module

        self.add_module('block3_conv1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.add_module('block3_relu1', _relu)  # reused ReLU module
        self.add_module('block3_conv2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.add_module('block3_relu2', _relu)  # reused ReLU module
        self.add_module('block3_conv3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.add_module('block3_relu3', _relu)  # reused ReLU module
        self.add_module('block3_maxpool', _maxpool)  # reused max_pooling module

        self.add_module('block4_conv1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block4_relu1', _relu)  # reused ReLU module
        self.add_module('block4_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block4_relu2', _relu)  # reused ReLU module
        self.add_module('block4_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block4_relu3', _relu)  # reused ReLU module
        self.add_module('block4_maxpool', _maxpool)  # reused max_pooling module

        self.add_module('block5_conv1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block5_relu1', _relu)  # reused ReLU module
        self.add_module('block5_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block5_relu2', _relu)  # reused ReLU module
        self.add_module('block5_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)) # --> 512 * W * H
        self.add_module('block5_relu3', _relu)  # reused ReLU module
        
        self.add_module('gmp', GMaxPool2d())  # global max_pooling module

        
        self.add_module('dropout', nn.Dropout(.5))  # dropout module
        self.add_module('dense', nn.Linear(512, 1))  # linear classifier expecting 512 inputs --> obtained by a global maxpooling over the features
        self.add_module('sigmoid', nn.Sigmoid())




class VGG16_Eyes(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _relu_inplace=True
        _relu = nn.ReLU(inplace=_relu_inplace)
        _maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.add_module('block1_conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.add_module('block1_relu1', _relu)  # reused ReLU module
        self.add_module('block1_conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.add_module('block1_relu2',_relu)  # reused ReLU module
        self.add_module('block1_maxpool',_maxpool)  # reused max_pooling module

        self.add_module('block2_conv1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))  
        self.add_module('block2_relu1',_relu)  # reused ReLU module
        self.add_module('block2_conv2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.add_module('block2_relu2',_relu)  # reused ReLU module
        self.add_module('block2_maxpool',_maxpool)  # reused max_pooling module

        self.add_module('block3_conv1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.add_module('block3_relu1', _relu)  # reused ReLU module
        self.add_module('block3_conv2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.add_module('block3_relu2', _relu)  # reused ReLU module
        self.add_module('block3_conv3', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.add_module('block3_relu3', _relu)  # reused ReLU module
        self.add_module('block3_maxpool',_maxpool)  # reused max_pooling module

        self.add_module('block4_conv1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block4_relu1', _relu)  # reused ReLU module
        self.add_module('block4_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block4_relu2', _relu)  # reused ReLU module
        self.add_module('block4_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block4_relu3', _relu)  # reused ReLU module
        self.add_module('block4_maxpool', _maxpool)  # reused max_pooling module

        self.add_module('block5_conv1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block5_relu1',_relu)  # reused ReLU module
        self.add_module('block5_conv2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.add_module('block5_relu2',_relu)  # reused ReLU module
        self.add_module('block5_conv3', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)) # --> 512 * W * H
        self.add_module('block5_relu3', _relu)  # reused ReLU module
        self.add_module('block5_maxpool', GMaxPool2d())  # global max_pooling module


class VGG16_Head(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_module('dropout', nn.Dropout(.5))  # dropout module
        self.add_module('dense', nn.Linear(512, 1))  # linear classifier expecting 512 inputs --> obtained by a global maxpooling over the features
