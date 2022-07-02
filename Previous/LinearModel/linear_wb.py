import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0,2.0,3.0,4.0]
y_data = [3.0,5.0,7.0,9.0]

#预测值
def forward(x,w,b):
    return x*w+b

#损失值  使用平方差之和求损失和
def loss(x,y,w,b):
    y_pred = forward(x,w,b)
    return (y_pred-y)*(y_pred-y)

wb_list = []
w_list = []
b_list = []
mse_list = []
x_list = []
y_list = []

#zip函数 将可迭代的对象作为参数打包为元组并返回元组列表
#a=[0,1,2]  b=[1,4,6]  zip(a,b) = [(0,1),(1,4),(2,6)]
w_list = np.arange(0.0,4.0,0.1)
b_list = np.arange(-1.0,3.0,0.1)
for w in w_list:
    for b in b_list:
        #print('w=',w,' b=',b)
        l_sum=0
        for x_val, y_val in zip(x_data,y_data):
            loss_val = loss(x_val,y_val,w,b)  #损失值
            l_sum += loss_val  #每组x,y 对应损失值的和
            #print('\t',x_val,y_val,y_pred_val,loss_val)
        x_num = len(x_data)
        MSEV = l_sum/x_num
        #print('MSE=',MSE)
        mse_list.append(MSEV)

MSE_list = np.array(mse_list).reshape(40,40)

w_list,b_list = np.meshgrid(w_list,b_list)  #生成网格点
#print(MSE_list)
fig = plt.figure()
ax = Axes3D(fig)


#绘制3d曲面
ax.plot_surface(w_list,b_list,MSE_list)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()

