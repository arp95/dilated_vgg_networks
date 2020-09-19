# header files
import numpy as np
import torch
import torch.nn as nn
import torchvision


class Convolution(torch.nn.Sequential):
    
    # init method
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, dilation=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation = dilation

        self.add_module("conv", torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.strides, padding=self.padding, dilation=self.dilation))
        self.add_module("norm", torch.nn.BatchNorm2d(self.out_channels))
        self.add_module("act", torch.nn.ReLU(inplace=True))
        

# define VGG19 network
class VGG19(nn.Module):
    
    # init method
    def __init__(self, num_classes = 2):
        super(VGG19, self).__init__()
        
        self.features = nn.Sequential(
            
            # first cnn block
            Convolution(3, 64, 3, 1, 1),
            Convolution(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # second cnn block
            Convolution(64, 128, 3, 1, 1),
            Convolution(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # third cnn block
            Convolution(128, 256, 3, 1, 1),
            Convolution(256, 256, 3, 1, 1),
            Convolution(256, 256, 3, 1, 1),
            Convolution(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # fourth cnn block
            Convolution(256, 512, 3, 1, 1),
            Convolution(512, 512, 3, 1, 1),
            Convolution(512, 512, 3, 1, 1),
            Convolution(512, 512, 3, 1, 1),
            
            # fifth cnn block
            Convolution(512, 512, 3, 1, 2, 2),
            Convolution(512, 512, 3, 1, 2, 2),
            Convolution(512, 512, 3, 1, 2, 2),
            Convolution(512, 512, 3, 1, 2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    # forward step
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
