import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 3,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        maxpool: int = 2,
        embedding_dim: int = 10,
        target_dim: int = 10
    ):
        """
        Basic Architecture of CNN

        Attributes:
            num_filters: Number of filters, out channel for 1st and 2nd conv layers,
            kernel_size: Kernel size of convolution,
            dense_layer: Dense layer units,
            img_rows: Height of input image,
            img_cols: Width of input image,
            maxpool: Max pooling size
            embedding_dim: the dimension of embedding layer
            target_dim: the dimension of target (i.e., the number of classes in the task)
        """
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 10, 2)
        self.b1=nn.BatchNorm2d(64)
        self.b2=nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 4)
        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = F.max_pool2d(x, 4)
        x = self.conv3(x)
       
        x = torch.flatten(x, 1)
        e = x.clone().detach()
        self.e=e
        
        
        return x



