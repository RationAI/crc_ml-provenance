"""
PyTorch models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as nn_func


class PretrainedVGG16(nn.Module):
    """
    VGG16 variant convolutional model.
    """

    def __init__(self):
        super().__init__()
        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block1_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block2_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block3_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block4_pool = nn.MaxPool2d(2, stride=2, padding=0)

        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.block5_pool = nn.MaxPool2d(2, stride=2, padding=0)

        self.dropout = nn.Dropout(.5)
        self.dense = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input to the model.

        Return
        ------
        torch.Tensor
            A tensor representing a binary decision over the input samples.
        """
        # Block 1
        x = nn_func.relu(self.block1_conv1(x))
        x = nn_func.relu(self.block1_conv2(x))
        x = self.block1_pool(x)

        # Block 2
        x = nn_func.relu(self.block2_conv1(x))
        x = nn_func.relu(self.block2_conv2(x))
        x = self.block2_pool(x)

        # Block 3
        x = nn_func.relu(self.block3_conv1(x))
        x = nn_func.relu(self.block3_conv2(x))
        x = nn_func.relu(self.block3_conv3(x))
        x = self.block3_pool(x)

        # Block 4
        x = nn_func.relu(self.block4_conv1(x))
        x = nn_func.relu(self.block4_conv2(x))
        x = nn_func.relu(self.block4_conv3(x))
        x = self.block4_pool(x)

        # Block 5
        x = nn_func.relu(self.block5_conv1(x))
        x = nn_func.relu(self.block5_conv2(x))
        x = nn_func.relu(self.block5_conv3(x))
        x = self.block5_pool(x)

        # GlobalMaxPool
        x = nn_func.max_pool2d(x, kernel_size=x.size()[2:])

        x = self.dropout(x)
        # [batch, side_size, 1, 1] -> [batch, 512, 1]
        x = torch.squeeze(x, -1)
        # [batch, side_size, 1] -> [batch, 512]
        x = torch.squeeze(x, -1)
        x = torch.sigmoid(self.dense(x))
        # [batch, 1] -> [batch]
        x = torch.squeeze(x, -1)

        return x
