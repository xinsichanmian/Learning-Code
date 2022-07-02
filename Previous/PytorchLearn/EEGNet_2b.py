# 导入工具包
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import scipy.io as sio
import torch.optim as optim
from Previous.PytorchLearn.LoadGraza2b import get_Xy,get_Xy2

class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(3, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T-1)
            nn.BatchNorm2d(16),  # output shape (16, 1, T-1)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T-1//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T-1//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,  #当groups与in_channels、out_channels相同时，是Depthwise Convolution
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(  #做1*1的卷积运算将Depthwise卷积运算的特征图进行加权融合，也便于扩展特征图即通道C
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear(496, classes_num)

    def forward(self, x):
        #print("block0", x.shape)
        x = self.block_1(x)
        #print("block1", x.shape)
        x = self.block_2(x)
        #print("block2", x.shape)
        x = self.block_3(x)
        #print("block3", x.shape)

        x = x.view(x.size(0), -1)
        #print("x.shape",x.shape)
        #print("view", x.shape)
        x = self.out(x)
        #print("fc",x.shape)
        #return F.softmax(x, dim=1), x  # return x for visualization
        return x

# data = sio.loadmat('dataset/BCICIV_2a_mat/A08.mat')
# x_train = data['train_x']
# y_train = data['train_y'][0]
x_train,y_train = get_Xy2()
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)

# y_t = torch.from_numpy(y_train[0:32])
# print(y_t)

#print(y_train)
batch_size = 32
samples = x_train.shape[0]
#print(samples)
model = EEGNet(2)

critizer = nn.CrossEntropyLoss()
optimier = optim.Adam(model.parameters(),lr=0.01)

# x1 = x_train[0:32]
# x1_t = torch.from_numpy(x1)
# print("shape1",x1_t.shape)
# x2 = x1_t.view(32,1,22,1125)
# print("shape2",x2.shape)
# print(x2)

for epoch in range(10):
    loss = 0.0
    correct = 0.0
    total = 0.0
    for i in range(samples//batch_size):
        s = i*batch_size
        e = (i+1)*batch_size
        #inputs = torch.from_numpy(x_train[s:e])
        inputs = torch.Tensor(x_train[s:e])
        inputs = inputs.view(batch_size, 1, 3, 1000)
        #print(inputs)

        #labels = torch.from_numpy(y_train[s:e])
        labels = torch.LongTensor(y_train[s:e])  #Label要求长整型
        outputs = model(inputs)
        # print("outputs",outputs)
        # print("labels",labels)

        _,idx = torch.max(outputs,dim=1)  #每一行的最大值即为预测结果
        correct += (idx==labels).sum().item()  #统计正确数量
        total += labels.size(0)

        optimier.zero_grad()
        loss = critizer(outputs,labels)
        loss.backward()
        optimier.step()
    print("epoch-%d loss %.3f Train accuracy %.3f%%"%(epoch,loss.item(),(correct/total)*100))
        #print("shape2",inputs.shape)

#保存模型参数
torch.save(model.state_dict(),"modelParameters/net2b.pth")
# if __name__ == '__main__':
#     input = torch.randn(32,1,22,1125)
#     model = EEGNet(2)
#     out = model(input)
#     print(model)
    #summary(model, (1, 1125))