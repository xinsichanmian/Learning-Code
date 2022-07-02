import numpy as np
import sklearn.naive_bayes as nb
import scipy.io as sio
from Previous.PytorchLearn.LoadGraza2b import get_Xy,get_Xy2

# 加载2b训练集，测试集
x_data, y_data = get_Xy2(dataset_dir="../dataset/BCICIV_2b_mat", sub="B01")
x_train = x_data[400:4700].reshape(4300, -1)  # 输入的数据集只能是二维，将EEG通道和采样时长合并为一个维度
y_train = y_data[400:4700]
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)

x_test = np.concatenate((x_data[:400], x_data[4700:])).reshape(884, -1)  # 输入的数据集只能是二维
y_test = np.concatenate((y_data[:400], y_data[4700:]))
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)

# clf = nb.GaussianNB()  #连续数据采用高斯贝叶斯分类
clf = nb.BernoulliNB()  #伯努利模型 伯努利模型适用于离散特征的情况，不同的是，伯努利模型中每个特征的取值只能是0和1。

clf.fit(x_train,y_train)

pre_train = clf.predict(x_train)
pre_test = clf.predict(x_test)
print("pre_test", pre_test)
print("y_test", y_test)

train_acc = (np.array(pre_train == y_train).astype(int)).sum() / y_train.shape[0]  # 2b训练集上的准确率
test_acc = (np.array(pre_test == y_test).astype(int)).sum() / y_test.shape[0]  # 2b测试集上的准确率
print("Train Accuracy %.3f%%" % (train_acc * 100))
print("Test Accuracy %.3f%%" % (test_acc * 100))

'''
                    2b BO1.mat二分类
                     GaussianNB       BernoulliNB       
        Train          64.744%          66.721%
        Test           52.262%          56.109%

'''