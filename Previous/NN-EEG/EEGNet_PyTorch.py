#!/usr/bin/env python
# coding: utf-8

# ## Utils 
# keras 2 Pytorch

# In[1]:


import torch
import torch.nn as nn



# In[2]:


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, kernel_size=3, stride=1, padding=0, bias=True):
        super(DepthwiseConv2D, self).__init__()
        self.DepthwiseConv2D = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                         kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=bias)

    def forward(self, input):
        out = self.DepthwiseConv2D(input)
        return out

# conv2D = DepthwiseConv2D(16, 32)
# summary(conv2D.cuda(), (16, 64, 64))


# In[3]:


class SeparableConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=True):
        super(SeparableConv2D, self).__init__()
        self.depthwiseConv2D = DepthwiseConv2D(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.point_conv = nn.Conv2d(
            in_channels=out_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, input):
        out = self.depthwiseConv2D(input)
        out = self.point_conv(out)
        return out

# model = SeparableConv2D(16, 32).cuda()
# summary(model, (16, 64, 64))


# In[4]:


from functools import reduce
from operator import __add__

def SamePadding( kernel_size = (4, 1)):
# Internal parameters used to reproduce Tensorflow "Same" padding.
# For some reasons, padding dimensions are reversed wrt kernel sizes,
# first comes width then height in the 2D case.
# https://stackoverflow.com/questions/58307036/is-there-really-no-padding-same-option-for-pytorchs-conv2d
    conv_padding = reduce(__add__, 
        [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    
    pad = nn.ZeroPad2d(conv_padding)
    return pad

# kernel_sizes = (4,3)
# conv = nn.Conv2d(1, 1, kernel_size=kernel_sizes)
# pad = SamePadding(kernel_size=kernel_sizes)

# x = torch.randn(size=(1, 1, 103, 40))
# print(x.shape) # (1, 1, 103, 40)
# print(conv(x).shape) # (1, 1, 100, 40)
# print(conv(pad(x)).shape) # (1, 1, 103, 40)


# # EEGNet 

# In[5]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ## My

# In[6]:


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8, D=2,
                 F2=16, norm_rate=0.25, dropoutType='Dropout'):
        """
        Inputs:

          nb_classes      : int, number of classes to classify
          Chans, Samples  : number of channels and time points in the EEGNet data
          dropoutRate     : dropout fraction
          kernLength      : length of temporal convolution in first layer. We found
                            that setting this to be half the sampling rate worked
                            well in practice. For the SMR dataset in particular
                            since the data was high-passed at 4Hz we used a kernel
                            length of 32.     
          F1, F2          : number of temporal filters (F1) and number of pointwise
                            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
          D               : number of spatial filters to learn within each temporal
                            convolution. Default: D = 2
          dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

        """
        super(EEGNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples

        # Layer 1
        kernel_size1 = (1, kernLength)
        self.block1 = nn.Sequential(
            SamePadding(kernel_size=kernel_size1),
            nn.Conv2d(1, F1, kernel_size1, bias=False),
            nn.BatchNorm2d(F1, False),
        )

        # Layer 2
        self.block2 = nn.Sequential(
            DepthwiseConv2D(F1, D*F1, kernel_size=(Chans, 1),
                            bias=False, padding=0),
            nn.BatchNorm2d(D*F1, False),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=dropoutRate),
        )

        # Layer 3
        kernel_size3 = (1, 16)
        self.block3 = nn.Sequential(
            SamePadding(kernel_size3),
            SeparableConv2D(D*F1, F2, kernel_size=kernel_size3,
                            bias=False, padding=0),
            nn.BatchNorm2d(F2, False),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropoutRate)
        )
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.

        features = F2 * (Samples // 4 // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, nb_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(-1, 1, self.Chans, self.Samples)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

#         # FC Layer
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    net = EEGNet(nb_classes=2, Chans=3,Samples=1000, kernLength=250).cuda(0)
    summary(net, (3, 1000))
