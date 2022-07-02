import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
'''
逻辑回归模型 与线性回归模型差异在
0.数据集  逻辑回归模型数据集 是分类  每个x对应的y是一个概率，概率总和为1
1.y_hat = σ（w*x+b）  而σ=0/0+e^-x  即 σ是逻辑函数  常被称为sigmoid函数
    故在forward函数中 计算y_pred = F.sigmoid(self.linear(x))
2.逻辑回归使用的二分类交叉熵(Binary Cross Entropy,BCE)作为损失函数 
'''

'''
pytroch写神经网络的步骤：
0.准备数据集
1.设计模型，每个模型都是一个类，继承torch.nn.Modeul 定义至少两个函数，构造函数__init__()和正向传播函数forward()
2.构造损失函数criterion（标准，准则）和优化器optimizer（优化器）
4.写训练周期，包括正向传播forward、反向传播backward、权重更新update
'''


# nn 是 neural network的缩写
# 所有的模型都以类的形式定义，且必须继承torch.nn.Module
class LogisticRegressionModel(torch.nn.Module):
    # 每个模型必须至少定义两个函数 构造函数init 和 正向传播函数forward
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # 定义 一个线性模型对象linear  类nn.Linear包含两个成员tensor，权重weight、偏移量bias
        '''
        torch.nn.Linear(in_features,out_features,bias=True)
        in_features 和 out_features是输入和输出的维度  bias=True要不要设置偏移量
        '''
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))  # 计算y_pred
        return y_pred
        '''
        在类对象后面加括号传参，实际上是调用的torch.nn.Linear类中的__call__()函数
        而在torch.nn.Module中，__call__函数中会调用forward()函数，所以在定义模型的时候必须定义forward函数
        举例讲解 __call__()函数
        class A:
            def __init__(self):
                pass
            def __call__(self,*args,**kwargs):
                print(args,kwargs)
                print("Hello"+str(args[0]))
            # args和kwargs是可变参数，直接传值，args是一个元组，而kwargs是一个词典

        a = A()
        a(0,1,2,4,x=6,y=7,z=8)  
        '''


'''
函数torch.nn.MSELoss(size_average=True,reduce=True)  均方差
size_average = True/False 求或不求损失平均值 
reduce 是否降维 
'''

'''
类 torch.optim.SGD(params,lr,...)
params 权重（如Linear中的w，b）
lr learing rate 学习率
'''

# 0.准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])  # 2*1的矩阵
y_data = torch.Tensor([[0], [0], [1]])  # 2*1的矩阵  逻辑回归的每个x对应一个概率y，用来分类的，

# 1.设计模型  包含上述LinearModel类的编写过程
logisticmodel = LogisticRegressionModel()

# 2.构造损失函数和优化器
# 构造损失函数 只需要y_pred和y就能算出损失总和或者损失平均值
criterion = torch.nn.BCELoss(size_average=False)  #逻辑回归的二分类交叉熵， 此处没求平均损失
# 构造优化器 有众多的优化器 优化器就是对权重进行更新的一个工具
# torch.optim.Adagrad、Adam、Adamax、ASGD、LBFGS、RMSprop、Rprop、SGD
optimizer = torch.optim.SGD(logisticmodel.parameters(), lr=0.01)  # 优化器对象  model.parameters()模型的权重

# 4.编写训练过程
# 训练过程：前馈、反馈、更新
# 四个步骤：0.算y_hat 1.根据y_hat和y计算损失（正向传播） 2.梯度的所有权重归零后反向传播  4.更新梯度的所有权重
for epoch in range(100):  # 定义100个循环周期
    y_pred = logisticmodel(x_data)  #得到y_pred也是一个矩阵
    loss = criterion(y_pred, y_data)  # 计算损失的过程就是构建计算图的过程
    print(epoch, loss)  # 打印loss时，会自动调用loss的__str__()函数
    optimizer.zero_grad()  # 让所有梯度的所有权重都归零后才能进行反向传播，即让上一次反向传播生成的梯度都归零

    loss.backward()  # 反向传播 计算梯度
    optimizer.step()  # 更新权重参数  如 w = w - lr*损失对w的偏导数

# 打印weight和bias
print('w=', logisticmodel.linear.weight.item())  # 如果tenser只有一个元素 .item()获取权重的值 否则就是一个weight的矩阵
print('b=', logisticmodel.linear.bias.item())  #

# 测试模型
x = np.linspace(0,10,200)  #采200个点  这是一个数组
print('x')
print(x)
x_t = torch.Tensor(x).view((200,1)) #将numpy数组变换成200 * 1的矩阵
y_t = logisticmodel(x_t)  #将x_t传入，调用的是模型类的__call__()函数，而__call__()函数中又调用了forward()函数，故logisticmodel(x_t)就调用的是forward()来计算y_t
y = y_t.data.numpy()  #将一个tensor变换成numpy数组
# print(x)
# print(y_t)
# print(y)
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hour')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
