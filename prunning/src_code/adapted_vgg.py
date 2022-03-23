import torch
import torch.nn as nn
import torch.nn.functional as F

from prunning.src_code import pruning_layer



class AdaptedVGG16(nn.Module):
    def __init__(self, model_weights=None):
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

        #self.fc1 = nn.Linear(512*7*7, 4096)
        #self.fc2 = nn.Linear(4096, 4096)
        #self.fc3 = nn.Linear(4096, 10)
        self.dropout = nn.Dropout(.5)  # dropout module
        self.dense = nn.Linear(512, 1)  # linear classifier expecting 512 inputs --> obtained by a global maxpooling over the features

        if model_weights is not None:
            #model_weight = torch.load(model_path)
            my_weight = self.state_dict()
            my_keys = list(my_weight.keys())
            count = 0
            for k, v in model_weights.items():
                print("Loading parameter", k, "with values", v.size())
                my_weight[my_keys[count]] = v
                count += 1
            self.load_state_dict(my_weight)


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
        x = self.maxpool(x)               # --> 512 * W/32 * H/32
        
        #print("SHAPE X:", x.size())
        # Global max pooling --> from each filter response take the maximum value (positive signal)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])  # results in a vector of size 512 * 1 * 1
        #print("SHAPE X2:", x.size())
        x = self.dropout(x)
        x = x.squeeze()
        #print("SHAPE X3:", x.size())
        #print("SHAPE Dense:", self.dense.weight.size())
        x = self.dense(x)
        #print("SHAPE X4:", x.size())
        x = torch.sigmoid(x)
    
        return x


class OnePrunableLayerAdaptedVGG16(torch.nn.Module):
    
    def __init__(self, layer_id:int, model_weight:dict):
        super().__init__()
        #torch.nn.Module.__init__(self)
        #model_weight = torch.load('checkpoint/model.pth')

        # at this point we might already be loading up a pruned model,
        # so we need to take the lengths from the model we load up
        channel_depths = list()  # tracking the channel depths (number on filters in a previus layer)
        channel_depths.append(3)  # no previous filters here, we just expect it's 3 (RGB picture)

        # collect channel depths (filter counts)
        for k, v in model_weight.items():
            if 'bias' in k:
                channel_depths.append(v.size()[0])

        self.feature_1 = nn.Sequential()  # all the layers up to the pruned one
        self.feature_2 = nn.Sequential()  # all the layers after the pruned one
        self.classifier = nn.Sequential()  # obligatory classifier at the end

        # TODO: Redefine the model as a sequence of prunable and nonprunable layers and use double indexing
        # index among prunable layers and index among all layers
        # layer names for each layer in the sequential model we are using (hardcoded)
        # TODO: Make it an input for the model
        conv_names = {0: 'block1_conv1', 1: 'block1_conv2', 2: 'block2_conv1', 3: 'block2_conv2', 4: 'block3_conv1', 5: 'block3_conv2', 6: 'block3_conv3',
                      7: 'block4_conv1', 8: 'block4_conv2', 9: 'block4_conv3', 10: 'block5_conv1', 11: 'block5_conv2', 12: 'block5_conv3'}
        # relu names are useess
        #relu_names = {0: 'block1_relu1', 1: 'block1_relu2', 2: 'block2_relu1', 3: 'block2_relu2', 4: 'block3_relu1', 5: 'block3_relu2', 6: 'block3_relu3',
        #              7: 'block4_relu1', 8: 'block4_relu2', 9: 'block4_relu3', 10: 'block5_relu1', 11: 'block5_relu2', 12: 'block5_relu3'}
        # pool names are also useless
        pool_names = {1: 'pool1', 3: 'pool2', 6: 'pool3', 9: 'pool4', 12: 'pool5'}
        #pooling_layer_id = {1, 3, 6, 9, 12}  # TODO remove,
        
        
        #ks_dict = {0: 224, 1: 224, 2: 112, 3: 112, 4: 56, 5: 56, 6: 56, 7: 28, 8: 28, 9: 28, 10: 14, 11: 14, 12: 14}
        # feature sizes for specific layers
        ks_dict = {0: 512, 1: 512, 2: 256, 3: 256, 4: 128, 5: 128, 6: 128, 7: 64, 8: 64, 9: 64, 10: 32, 11: 32, 12: 32}
        
        #print("H: channel_depth")
        print(ks_dict.values())
        print(model_weight.keys())
        print(channel_depths)

        # add channel selection layer
        # define the autopruner layer respecting the channel depth
        self.CS = pruning_layer.MyCS(channel_depths[layer_id+1], activation_size=ks_dict[layer_id], kernel_size=2)
        
        

        # add feature_1 and feature_2 layers
        current_sequential_model = self.feature_1  # start adding layers to the first half
        for i in range(13): 
            # add a convolutional layer with predefined name (has to match the name from the previous model)
            current_sequential_model.add_module(
                conv_names[i], nn.Conv2d(channel_depths[i], channel_depths[i + 1], kernel_size=3,  stride=1, padding=1))
            # add ReLU
            current_sequential_model.add_module(f"relu{i}", nn.ReLU(inplace=True))

            # if we are at the pruned layer, switch the sequential Module we are appending to to the next one
            if i == layer_id:  
                current_sequential_model = self.feature_2

            # if there is a pooling layer expected, add a pooling layer
            if i in pool_names:
                current_sequential_model.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
            
            

        # TODO: Generalize the method for use with any sequential model. Any number of pruned layers.
        #for i in range(13): 
            
            #if i < layer_id:
                
                #self.feature_1.add_module(conv_names[i],
                #                          nn.Conv2d(channel_depths[i], channel_depths[i + 1], kernel_size=3, stride=1,
                #                                    padding=1))
                #self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
                #
                #if i in pool_names:
                #    self.feature_1.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))

            #elif i == layer_id:
            
                # self.feature_1.add_module(conv_names[i],
                #                           nn.Conv2d(channel_depths[i], channel_depths[i + 1], kernel_size=3, stride=1,
                #                                     padding=1))
                # self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
                # if i in pool_names:
                #     self.feature_2.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
            # elif i > layer_id:
            #     self.feature_2.add_module(conv_names[i],
            #                               nn.Conv2d(channel_depths[i], channel_depths[i + 1], kernel_size=3, stride=1,
            #                                         padding=1))
            #     self.feature_2.add_module(relu_names[i], nn.ReLU(inplace=True))
            #     if i in pool_names:
            #         self.feature_2.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
            
            
           

        # add classifier
        # self.classifier.add_module('fc6', nn.Linear(channel_depths[13] * 7 * 7, channel_depths[14]))
        # self.classifier.add_module('relu6', nn.ReLU(inplace=True))
        # self.classifier.add_module('dropout6', nn.Dropout())

        # self.classifier.add_module('fc7', nn.Linear(channel_depths[14], channel_depths[15]))
        # self.classifier.add_module('relu7', nn.ReLU(inplace=True))
        # self.classifier.add_module('dropout7', nn.Dropout())

        # self.classifier.add_module('fc8', nn.Linear(channel_depths[15], channel_depths[16]))

        self.classifier.add_module('dropout', nn.Dropout(.5)) # dropout module
        self.classifier.add_module('dense', nn.Linear(512, 1)) # fully connected layer
        self.classifier.add_module('sigmoid', nn.Sigmoid()) # sigmoid at the end (for CrossEntropy)


        # load pretrain model weights
        my_weight = self.state_dict()  # get the reference on the state_dict of this model
        my_keys = list(my_weight.keys())  # get the parameter names
        #print("MY_WEIGHTS:", my_weight.keys())
        # loop over the parameters from the weights being loaded
        for k, v in model_weight.items():
            print("NAME:", k)
            name = k.split('.')
            name = 'feature_1.'+name[0]+'.'+name[1]
            if name in my_keys:
                my_weight[name] = v

            name = k.split('.')
            name = 'feature_2.' + name[0] + '.' + name[1]
            if name in my_keys:
                my_weight[name] = v

            name = k#[7:]
            if name in my_keys:
                my_weight[name] = v
        self.load_state_dict(my_weight)

    def forward(self, x, scale_factor=1.0, channel_index=None):
        x = self.feature_1(x)
        x, scale_vector = self.CS(x, scale_factor, channel_index)
        x = self.feature_2(x)
        
        # Global Max Pooling - dependent on the input size (easier to define here like this)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])  # results in a vector of size 512 * 1 * 1
        #print("F-Extr res:", x.shape)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x, scale_vector

        