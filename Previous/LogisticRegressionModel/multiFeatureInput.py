import numpy as np
import torch

#加载糖尿病数据 这是个8维的输入，即有8列  是个n*8的矩阵 选择32位的浮点数是因为大部分显卡GPU支持的是32位计算
xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)  #直接加载称为一个n*9的矩阵  最后一列是y的值
#print(xy[:,[-0]])  #第一个冒号表示行， 加载全部行，第二个冒号代表列，加载从第1行至倒数-1列，即倒数第二列
x_data = torch.from_numpy(xy[:,:-1])   #第一个冒号表示行， 加载全部行，第二个冒号代表列，加载从第1行至倒数-1列，即倒数第二列
y_data = torch.from_numpy(xy[:,[-1]])  #-1列加中括号是 让该列用矩阵的形式表示，否则就只有一个数据没有括号


#设计模型 第一层 8*6 的线性模型，第二层6*4的线性模型，第三层4*1的线性模型
# 模型并不是层数越多越好，越多学习能力越强，泛化能力可能就比较弱
class MultiFeatureInputModel(torch.nn.Module):
    def __init__(self):
        super(MultiFeatureInputModel,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)  #第一层，输入特征8个维度，输出特征6个维度
        self.linear2 = torch.nn.Linear(6,4)  #第二层，输入特征6个维度，输出特征4个维度
        self.linear3 = torch.nn.Linear(4,1)  #第三层，输入特征4个维度，输出特征1个维度
        self.sigmoid = torch.nn.Sigmoid()    #在这儿定义一个sigmoid函数，因为三层均需要使用激活函数，所以就当作一个属性直接用

        #注意：  如果使用的激活函数是torch.nn.ReLu，因为ReLu是将小于或等于0的输入全输出为0，所以最终输出y有可能为0
        #有可能在求损失的时候会使用logy，为了避免y为0，在前几层的计算中使用ReLu作为损失函数
        # 在forward函数最后一层计算y的值时，使用一个sigmoid函数（逻辑函数）就能避免最终输出y的值是0

    #一般我们就用一个变量x，在中间层，x就是上一层的输出y,也是下一层的输入x，所以我们就用一个x，连环使用，最后得到x就是计算出的y
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))   #计算第一层的输出结果 由输入的n*8 tensor 变成输出的n*6 tensor
        x = self.sigmoid(self.linear2(x))   #计算第二层的输出结果 由n*6 变成 n*4
        x = self.sigmoid(self.linear3(x))   #计算第三层的输出结果  由n*4 变成 n*0 实现了由特征8维输入变特征1维输出
        return x
        ''' 
        因为torch.nn.Linear类继承了torch.nn.Module类，父类里定义了__call__()函数，
        而torch.nn.Module类中的__call__()函数调用forward()函数，且返回结果就是调用forward()函数的结果
        所以可以直接用Linear类对象linear加括号传入参数linear(x),调用的就是__call__()函数，间接调用forward函数计算y_pred的值
        '''

multimodel = MultiFeatureInputModel()

#构造损失函数，因为在这儿还是逻辑回归模型，只是输入维度不是1维，而是多维，所以依旧用BCEloss
criterion = torch.nn.BCELoss(size_average=True)

#构造优化器
optimizer = torch.optim.SGD(multimodel.parameters(),lr=0.01)  #传入模型的权重参数，设置学习率lr

#编写训练周期  设置100个周期
for epoch in range(0,100):
    y_pred = multimodel(x_data)  #计算y_pred，x_data是tensor 那么计算得到的y_pred也是tensor
    loss = criterion(y_pred,y_data)  #计算损失
    print("y_pred_shpae",y_pred.shape)
    print("y_data_shape",y_data.shape)
    print(epoch,loss.item())  #打印训练周期和损失值

    optimizer.zero_grad() #梯度归零
    loss.backward()  #反向传播，计算梯度
    optimizer.step()  #更新权重参数 w=w-lr*grad

list = [0.15,0.83,0.23,-0.02,0.0,0.31,0.53,0.14]
x_test = torch.Tensor([[0.15,0.83,0.23,-0.02,0.0,0.31,0.53,0.14],[0.41,-0.21,0.24,-0.08,0.0,-0.21,0.62,-0.43]])
y_test = multimodel(x_test)
print('测试结果',y_test.data)
