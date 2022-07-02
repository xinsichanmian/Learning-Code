import scipy.io as sio
from sklearn import svm
import numpy as np
from Previous.PytorchLearn.LoadGraza2b import get_Xy,get_Xy2

#加载2b训练集，测试集
x_data,y_data = get_Xy2(dataset_dir="../dataset/BCICIV_2b_mat", sub="B01")
x_train = x_data[400:4700].reshape(4300,-1)  #输入的数据集只能是二维
y_train = y_data[400:4700]
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)

x_test = np.concatenate((x_data[:400],x_data[4700:])).reshape(884,-1)
y_test = np.concatenate((y_data[:400],y_data[4700:]))
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)

# #加载2a训练集、测试集
# data = sio.loadmat("../dataset/BCICIV_2a_mat/A01.mat")
# d_k1 = data.keys()
# print(d_k1)
#
# x_train = data['train_x']
# y_train = data['train_y'][0]
# x_train = x_train.reshape(576,-1)
# print("x_train.shape",x_train.shape)
# print("y_train.shape",y_train.shape)
#
# #加载训练集
# data = sio.loadmat("../dataset/BCICIV_2a_mat/A02.mat")
# x_test = data['train_x']
# y_test = data['train_y'][0]
# x_test = x_test.reshape(576,-1)
# print("x_train.shape",x_test.shape)
# print("y_train.shape",y_test.shape)

#clf = svm.LinearSVC()
#clf = svm.SVC()
#clf = svm.NuSVC()
clf = svm.SVC(kernel='poly')
#clf = svm.SVR()
clf.fit(x_train,y_train)
print("Train sucessful")
pre_train = clf.predict(x_train)
pre_test = clf.predict(x_test)
print("pre_test",pre_test)
print("y_test",y_test)
# print("pre_test.shape",pre_test.shape)
#print("label ",type(pre_test))
# pre_test1 = (pre_train == y_train)

train_acc = (np.array(pre_train == y_train).astype(int)).sum() / y_train.shape[0]  #2b训练集上的准确率
test_acc = (np.array(pre_test==y_test).astype(int)).sum() / y_test.shape[0]     #2b测试集上的准确率

# train_acc = (pre_train == y_train).sum() / y_train.shape[0]  #2a训练集上的准确率
# test_acc = (pre_test==y_test).sum() / y_test.shape[0]     #2a测试集上的准确率
#print("Test Accuracy %.3f%%" % (clf.score(x_test,y_test))%100)
print("Train Accuracy %.3f%%" % (train_acc*100))
print("Test Accuracy %.3f%%" % (test_acc*100))

'''
SVC中的参数C越大，对于训练集来说，其误差越小，但是很容易发生过拟合；C越小，则允许有更多的训练集误分类，相当于soft margin
# 选择 SVM的训练容器
线性分类             2a_A02.mat四分类                        2b_BO1T.mat二分类
LinearSVC Test Acc  rbf:28.472%                         rbf:Train 68%   Test 51.810%
SVC                 rbf:32.465%  poly:27.431%           rbf:Train 100%  Test 48.077%   poly:Train 100%  Test 43.778%
NuSVC               rbf:32.639%  poly:27.083%           rbf:Train 100%  Test 45.136%

非线性分类 逻辑回归 
'''

#
# print(clf.support_vectors_)  # 支持向量点
#
# print(clf.support_)  # 支持向量点的索引
#
# print(clf.n_support_)  # 每个class有几个支持向量点

# x = np.array([[[2, 0, 1]], [[1, 1, 2]], [[2, 3, 3]]])
# print("x",x.shape)
# x = x.reshape(3,-1)
# print(x)
# y = np.array([0, 0, 1])   # 分类标记
# clf = svm.SVC(kernel='linear')  # SVM模块，svc,线性核函数
# clf.fit(x, y)
#
# print(clf)
#
# print(clf.support_vectors_)  # 支持向量点
#
# print(clf.support_)  # 支持向量点的索引
#
# print(clf.n_support_)  # 每个class有几个支持向量点

# print(clf.predict([[2, 0, 3]]))  # 预测