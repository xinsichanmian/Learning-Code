import scipy.io as sio
import numpy as np
import torch

data = sio.loadmat("dataset/BCICIV_2a_mat/A01.mat")
d_k1 = data.keys()
print(d_k1)
x_train = data['train_x']
y_train = data['train_y'][0]
print("x_train.shape",x_train.shape)
print("y_train.shape",y_train.shape)
for i in range(2,5):
    data = sio.loadmat("dataset/BCICIV_2a_mat/A0"+str(i)+".mat")
    x_train = np.vstack((x_train,data['train_x']))
    y_train = np.concatenate((y_train,data['train_y'][0]))
    print("x_train.shape",x_train.shape)
    print("y_train.shape",y_train.shape)

# print(train_y[0])
# #
# # #print(data)
# # x_t = torch.Tensor(train_x)
# # y_t = torch.Tensor(train_y)
# print("x_train.shape",train_x.shape)
# print("y_train.shape",train_y.shape)
# print(train_x)
#
# data = sio.loadmat("dataset/BCICIV_2b_mat/B01E.mat")
# d_k = data.keys()
# print(d_k)
# data2 = sio.loadmat("dataset/BCICIV_2b_mat/B01T.mat")
# d_k2 = data2.keys()
# print(d_k2)
# x_data1 = data['data']
# x_data2 = data2['data']
# print(x_data2.shape)
# #print(x_data2)
# #print(x_data2)
# # # print(x_data2.shape)
# #print(x_data2[0])
# # print("+++++++++++++++++++++++++++")
# # print("+++++++++++++++++++++++++++")
# x_data2_1 = x_data2[0][0]
# print("x_data2_1.shape",x_data2_1.shape)
# #print(x_data2_1)
# print("+++++++++++++++++++++++++++++++")
# # print("+++++++++++++++++++++++++++")
# x_data2_1_1 = x_data2_1[0][0]
# print("x_data2_1_1.shape",len(x_data2_1_1))
# #print(x_data2_1_1)
#
# # print("+++++++++++++++++++++++++++")
# # print("+++++++++++++++++++++++++++++++")
# x_data2_1_1_1 = x_data2_1_1[0]
# x_data2_1_1_2 = x_data2_1_1[1]
# x_data2_1_1_3 = x_data2_1_1[2]
# x_data2_1_1_5 = x_data2_1_1[5]
# # print(len(x_data2_1_1))
# print("x_data2_1_1_1.shape",x_data2_1_1_1.shape)
# print("x_data2_1_1_2.shape",x_data2_1_1_2.shape)
# print("x_data2_1_1_3.shape",x_data2_1_1_3.shape)
# print("x_data2_1_1_5.shape",x_data2_1_1_5.shape)
# #print(x_data2_1_1_2)
# #print(x_data2_1_1)
# #print(x_data2_1_1_1)
# # for t in x_data2_1_1:
# #     print(t)
# #     print("--------------------------")
#
# # print(data)
#
# data3 = sio.loadmat("dataset/true_labels2b/B0102T.mat")
# d_k1 = data3.keys()
# print(d_k1)
# y_d = data3['classlabel']
# print("y_d.shape",y_d.shape)
#
# #print(y_d)
# #print(x_data2_1_1)
# print("equal2",(x_data2_1_1_2==y_d).sum())
# print("equal3",(x_data2_1_1_3==y_d).sum())
# print("equal5",(x_data2_1_1_5==y_d).sum())