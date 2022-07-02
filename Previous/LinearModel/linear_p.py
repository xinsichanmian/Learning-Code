import torch
import numpy as np

'''
pytroch写神经网络的步骤：
0.准备数据集
1.设计模型，每个模型都是一个类，继承torch.nn.Modeul 定义至少两个函数，构造函数__init__()和正向传播函数forward()
2.构造损失函数criterion（标准，准则）和优化器optimizer（优化器）
4.写训练周期，包括正向传播forward、反向传播backward、权重更新update
'''

# nn 是 neural network的缩写
# 所有的模型都以类的形式定义，且必须继承torch.nn.Module
class LinearModel(torch.nn.Module):
    # 每个模型必须至少定义两个函数 构造函数init 和 正向传播函数forward
    def __init__(self):
        super(LinearModel, self).__init__()
        # 定义 一个线性模型对象linear  类nn.Linear包含两个成员tensor，权重weight、偏移量bias
        '''
        torch.nn.Linear(in_features,out_features,bias=True)
        in_features 和 out_features是输入和输出的维度  bias=True要不要设置偏移量
        '''
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)  #计算y_pred
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

#0.准备数据集
x_data = torch.Tensor([[1.0],[2.0],[3.0]])  #2*1的矩阵
y_data = torch.Tensor([[2.0],[4.0],[6.0]])  #2*1的矩阵

#1.设计模型  包含上述LinearModel类的编写过程
model = LinearModel()

#2.构造损失函数和优化器
#构造损失函数 只需要y_pred和y就能算出损失总和或者损失平均值
criterion = torch.nn.MSELoss(size_average=False)
#构造优化器 有众多的优化器 优化器就是对权重进行更新的一个工具
#torch.optim.Adagrad、Adam、Adamax、ASGD、LBFGS、RMSprop、Rprop、SGD
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)  #优化器对象  model.parameters()模型的权重

#4.编写训练过程
#训练过程：前馈、反馈、更新
# 四个步骤：0.算y_hat 1.根据y_hat和y计算损失（正向传播） 2.梯度的所有权重归零后反向传播  4.更新梯度的所有权重
for epoch in range(100):  #定义100个循环周期
    y_pred = model(x_data)  #计算y_pred 调用的是LinearModel类中的__call__()，而__call__又调用了forward() 故类对象传参数model(x)就调用的是forward()
    loss = criterion(y_pred,y_data)  #计算损失的过程就是构建计算图的过程
    print(epoch,loss)  #打印loss时，会自动调用loss的__str__()函数
    optimizer.zero_grad()  #让所有梯度的所有权重都归零后才能进行反向传播，即让上一次反向传播生成的梯度都归零

    loss.backward() #反向传播 计算梯度
    optimizer.step()  #更新梯度  如 w = w - lr*损失对w的偏导数

#打印weight和bias
print('w=',model.linear.weight.item())  # 如果tenser只有一个元素 .item()获取权重的值 否则就是一个weight的矩阵
print('b=',model.linear.bias.item())  #

#测试模型
x_test = torch.Tensor([[4.0],[10.0]])  #这是一个2*1的矩阵
y_test = model(x_test)
'''
LinearModel继承的是torch.nn.Module,而父类定义了__call__ ()函数，故LinearModul也继承了__call__函数()
torch.nn.Module类中__call__()调用了forward()，__call__()返回结果就是调用的forward函数的值
所以LinearModel类对象model在括号后传入x_test就相当于调用的是forward()函数来计算y_test
'''
print('y_pred = ',y_test.data)  #根据线性模型输入是2*1的矩阵，那么得到的也是2*1的矩阵
