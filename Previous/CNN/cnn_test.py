import torch

import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ),(0.3081, ))
])

batch_size = 64
#0.读取数据
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True, #是训练集
                               download=True,
                               transform=transform) #按照我们设置的转变将图片进行转变
train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True)
# for i in train_dataset:
#     print(i)

for data in train_loader:
    inputs,label = data
    print("input",inputs.shape,type(inputs))

    print("label",label)

# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1,
#          ]
#
#
# #参数顺序 小批量Bach_size=0  通道Channel=0， 宽度W=5， 高度H=5
# input = torch.Tensor(input).view(1,1,5,5)
#
# #参数顺序 输入通道input_Channel=0， 输出通道out_Channel=0，卷积核尺寸3*2 填充padding=0 步长stride=0 偏移量bias=False
# conv_layer = torch.nn.Conv2d(1,1,kernel_size=3,padding=1,stride=1,bias=False)
# #填充之后 输入就变成了 0*7*7的输入
#
# #参数顺序 输出通道数1，输入通道数，卷积核宽3  高 2 即3*2
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
# print("kernel",kernel.data)
# print("conv_layer中的kernel赋值——前",conv_layer.weight.data)
# #将自定义的卷积核中权重值的值赋予给卷积层conv_layer的卷积核的权重值，即自己初始化卷积层的权重
# conv_layer.weight.data = kernel.data
# print("conv_layer中的kernel赋值——后",conv_layer.weight.data)
# output = conv_layer(input)
# print(output)  #可以使用output.data来获取张量的值（矩阵）


