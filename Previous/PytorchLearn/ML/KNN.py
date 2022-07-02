from sklearn.datasets.base import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Previous.PytorchLearn.LoadGraza2b import get_Xy2,get_Xy
import scipy.io as sio

def knn_2b():
    #加载2b训练集，测试集
    x_data,y_data = get_Xy2(dataset_dir="../dataset/BCICIV_2b_mat", sub="B01")
    x_train = x_data[400:4700].reshape(4300,-1)  #输入的数据集只能是二维，将EEG通道和采样时长合并为一个维度
    y_train = y_data[400:4700]
    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)

    x_test = np.concatenate((x_data[:400],x_data[4700:])).reshape(884,-1)   #输入的数据集只能是二维
    y_test = np.concatenate((y_data[:400],y_data[4700:]))
    print("x_test.shape", x_test.shape)
    print("y_test.shape", y_test.shape)

    knn = KNeighborsClassifier()  #分类器
    knn.fit(x_train,y_train)    #拟合训练分类
    print("Train Sucessful")

    pre_train = knn.predict(x_train)
    pre_test = knn.predict(x_test)
    print("pre_test",pre_test)
    print("y_test",y_test)

    train_acc = (np.array(pre_train == y_train).astype(int)).sum() / y_train.shape[0]  #2b训练集上的准确率
    test_acc = (np.array(pre_test==y_test).astype(int)).sum() / y_test.shape[0]     #2b测试集上的准确率
    print("Train Accuracy %.3f%%" % (train_acc*100))
    print("Test Accuracy %.3f%%" % (test_acc*100))

def knn_2a():
    # 加载2a训练集
    data = sio.loadmat("../dataset/BCICIV_2a_mat/A01.mat")
    # d_k1 = data.keys()
    # print(d_k1)
    x_train = data['train_x']
    y_train = data['train_y'][0]
    for i in range(2, 8):
        data = sio.loadmat("../dataset/BCICIV_2a_mat/A0" + str(i) + ".mat")
        x_train = np.vstack((x_train, data['train_x']))
        y_train = np.concatenate((y_train, data['train_y'][0]))
    x_train = x_train.reshape(4032,-1)  #压缩为二维输入，KNN只支持2个维度的数据集 (4032,22,1125)
    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)

    # 加载测试集
    data = sio.loadmat("../dataset/BCICIV_2a_mat/A08.mat")
    x_test = data['train_x']
    y_test = data['train_y'][0]
    data = sio.loadmat("../dataset/BCICIV_2a_mat/A09.mat")
    x_test = np.vstack((x_test, data['train_x']))
    x_test = x_test.reshape(1152,-1)  #(1152,22,1125) 三维变为二维输入
    y_test = np.concatenate((y_test, data['train_y'][0]))
    print("x_test.shape", x_test.shape)
    print("y_test.shape", y_test.shape)

    knn = KNeighborsClassifier()
    knn.fit(x_train,y_train)
    print("Train Finish")
    pre_train = knn.predict(x_train)
    pre_test = knn.predict(x_test)
    # print("pre_test", pre_test)
    # print("y_test", y_test)
    train_acc = (pre_train == y_train).sum() / y_train.shape[0]  #2a训练集上的准确率
    test_acc = (pre_test==y_test).sum() / y_test.shape[0]     #2a测试集上的准确率
    print("Train Accuracy %.3f%%" % (train_acc*100))
    print("Test Accuracy %.3f%%" % (test_acc*100))
    #print("Test Accuracy %.3f%%" % (knn.score(x_test,y_test))%100)

knn_2a()

'''
KNN分类
                     2b BO1T.mat 二分类        2a A01T.mat 四分类     2a A01T-A107T  A07-9T.mat 四分类
Train Accuracy          64.047%                     62.326%                 
Test Accuracy           46.946%                     26.389%
'''


# # 从 sklearn的datasets模块载入数据集加载酒的数据集
# wineDataSet = load_wine()
# print(wineDataSet)
# print("红酒数据集中的键：\n{}".format(wineDataSet.keys()))
# print("数据概况：\n{}".format(wineDataSet['data'].shape))
# print(wineDataSet['DESCR'])
#
# # 将数据集拆分为训练数据集和测试数据集
# X_train, X_test, y_train, y_test = train_test_split(wineDataSet['data'], wineDataSet['target'], random_state=0)
# print("X_train shape:{}".format(X_train.shape))
# print("X_test shape:{}".format(X_test.shape))
# print("y_train shape:{}".format(y_train.shape))
# print("y_test shape:{}".format(y_test.shape))
#
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# print(knn)
#
# # 评估模型的准确率
# print('测试数据集得分：{:.2f}'.format(knn.score(X_test, y_test)))
#
# # 使用建好的模型对新酒进行分类预测
# X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
# prediction = knn.predict(X_new)
# print("预测新酒的分类为：{}".format(wineDataSet['target_names'][prediction]))
