import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#预测值
def forward(x,w):
    return x*w

#损失值  使用平方差之和求损失和
def loss(x,y,w):
    y_pred = forward(x,w)
    return (y_pred-y)*(y_pred-y)

w_list = []
mse_list = []

#zip函数 将可迭代的对象作为参数打包为元组并返回元组列表
#a=[0,1,2]  b=[1,4,6]  zip(a,b) = [(0,1),(1,4),(2,6)]

for w in np.arange(0.0,4.0,0.1):
    print('w=',w)
    l_sum=0
    for x_val, y_val in zip(x_data,y_data):
        y_pred_val = forward(x_val,w)  #预测值
        loss_val = loss(x_val,y_val,w)  #损失值
        l_sum += loss_val  #每组x,y 对应损失值的和
        print('\t',x_val,y_val,y_pred_val,loss_val)
    x_num = len(x_data)
    MSE = l_sum/x_num
    print('MSE=',MSE)
    w_list.append(w)
    mse_list.append(MSE)

plt.plot(w_list,mse_list)
plt.ylabel('LOSS')
plt.xlabel('w')
plt.show()