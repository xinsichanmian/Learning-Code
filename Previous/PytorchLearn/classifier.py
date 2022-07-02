import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),  # 随机裁剪将图片裁剪为224*224
    # transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageFolder root：在root指定的路径下寻找图片
# transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
# target_transform：对label的转换
# loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象

train_path = 'dataset/train_data/'
test_path = 'dataset/test_data/'
train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)
# print(dataset.imgs)  #所有图片的路径和对应的label
# print(dataset.class_to_idx)  #分类和对应标签
print(train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  #label  张量(batch_size)

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# for input,label in train_loader:
#     print(label)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 5)
        self.avgpl = nn.AvgPool2d(2)  # 平均池化 2*10*110*110
        self.conv2 = nn.Conv2d(10, 20, 5)  # 2*20*106*106
        self.conv3 = nn.Conv2d(20,10,5)  # 2*10*102
        self.fc = nn.Linear(5760,3)  #变成3分类

    def forward(self, x):
        batch_size = x.size(0)  # 获取batch_size
        # 输入的是 2*3*224*224 经过第一层卷积和平均池化
        x = F.relu(self.avgpl(self.conv1(x)))  # RGB图像是3通道 2*10*110
        #print("conv1", x.size())
        x = F.relu(self.avgpl(self.conv2(x)))  # 2*20*53*53
        #print("conv2", x.size())
        x = F.relu(self.avgpl(self.conv3(x)))   #2*10*49*49 -> 2*10*24*24
        x = x.view(batch_size, -1)  # 计算输入FC的维度
        #print("x", x.size(1))
        x = self.fc(x)
        return x


model = CNNClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    for idx, data in enumerate(train_loader, 0):
        input, label = data
        output = model(input)
        optimizer.zero_grad()
        loss = criterion(output, label)
        print("output",output)
        print("label",label)
        loss.backward()
        optimizer.step()
        #print("label", label)
        #print("output", output.shape)
    print("Epoch %d  loss %.3f"%(epoch+1,loss.item()))

def test(epoch):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            input,label = data
            print(label)
            output = model(input)
            total += input.size(0)  # .size(0)  batch_size
            _, maxidx = torch.max(output, dim=1)  # 寻找行最大值的下标（最大概率就是分类预测的结果）
            # print("maxidx", maxidx)
            # print("output",output)
            # print("label",label)
            correct += (label == maxidx).sum().item()  # 两个张量中值相等时新张量该值为1，然后求和取item()
    print("Epoch %d  Accuracy on test %0.3f %%" % (epoch+1,correct / total))


if __name__ == '__main__':
    for i in range(20):
        train(i)
        test(i)

