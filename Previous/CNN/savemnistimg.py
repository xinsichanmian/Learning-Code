import torch
import torchvision
import torch.utils.data as Data
import scipy.misc
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
BATCH_SIZE = 50
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(root='../dataset/mnist/', train=True ,transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST, )
test_data = torchvision.datasets.MNIST(root='../dataset/mnist/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), requires_grad=True).type(torch.FloatTensor)[:20 ] /255
test_y = test_data.test_labels[:20  ]  # 前两千张
# 具体查看图像形式为：
a_data, a_label = train_data[0]
print(type(a_data)  )  # tensor 类型
# print(a_data)
print(a_label)

# 把原始图片保存至MNIST_data/raw/下
save_dir ="mnist_imgs/"
if os.path.exists(save_dir )is False:
    os.makedirs(save_dir)
for i in range(20):
    image_array , _ =train_data[i  ]  # 打印第i个
image_array =image_array.resize(28 ,28)
filename =save_dir + 'mnist_train_%d.jpg' %  i  # 保存文件的格式
print(filename)
print(train_data.train_labels[i])  # 打印出标签
scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)  # 保存图像

