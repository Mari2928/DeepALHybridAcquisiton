import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(
        self,
        structure= [64, 'M', 128, 'M', 256, 'M', 512,'M', 10,'M']
    ): 
        super(ConvNN, self).__init__()
        self.structure=structure
        self.vgg=self.make_vgg()
        




    def make_vgg(self,batch_norm=True):
        layers = []
        in_channels = 3
        structure=self.structure
        for v in structure:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True),nn.Dropout(0.1)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg(x)
        x = torch.flatten(x, 1) 
        e = x.clone().detach()
        self.e=e
        
        
        return x





    




