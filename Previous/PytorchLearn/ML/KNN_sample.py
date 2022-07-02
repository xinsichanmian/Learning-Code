# 导入画图工具
import matplotlib.pyplot as plt
# 导入数组工具
import numpy as np
# 导入数据集生成器
from sklearn.datasets import make_blobs
# 导入KNN 分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成样本数为500，分类数为5的数据集
data = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=1.0, random_state=8)
X, Y = data
# print("X",X.shape)
print("X",Y)
# 将生成的数据集进行可视化
# plt.scatter(X[:,0], X[:,1],s=80, c=Y,  cmap=plt.cm.spring, edgecolors='k')
# plt.show()

clf = KNeighborsClassifier()
clf.fit(X, Y)

# 绘制图形
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y, cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:KNN")

# 把待分类的数据点用五星表示出来
plt.scatter(0, 5, marker='*', c='red', s=200)

# 对待分类的数据点的分类进行判断
res = clf.predict([[0, 5]])
plt.text(0.2, 4.6, 'Classification flag: ' + str(res))
plt.text(3.75, -13, 'Model accuracy: {:.2f}'.format(clf.score(X, Y)))

plt.show()
