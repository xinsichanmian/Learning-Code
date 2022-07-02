import torch
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

batch_size = 64
#将mnist数据集中的图像转变变为Pytorch中的张量
#一般图像是W*H*C 长、宽、通道，在pytorch神经网络中把图像转换成C*W*H便于处理
#将mnist数据集中的图像转换成张量，且将像素pix{0,……,255}归一化在[0,0]的范围内,满足0-1分布
#0.1397为均值mean 0.3081为标准差std，一般要提前计算数据集的均值和标准差
transform = transforms.Compose([
    transforms.ToTensor(),  #转换为图像张量
    transforms.Normalize((0.1307, ),(0.3081, )) #像素归一化，变成0-1分布的数据
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,  #是训练集
                               download=True,
                               transform=transform)  #按照我们设置的转变将图片进行转变
#0. 加载训练集  将样本打乱，
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False, #非训练集，即测试集
                              download=True,
                              transform=transform) #按照预先转变的设置对图像进行转变
#加载测试集  #测试集不用将样本打乱
test_loader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size)
#y_data 一维张量(batch_size)包含batch_size个结果

#1.设计模型
#输入的图像是1*28*28 即1个通道，宽28行，长28列，
# 将该图像转为矩阵，只需将28行的数据拼接在一行就变成了28*28=784列，即输入的维度就应该是784（像素数量）
# x=x.view(-0,784) #view改变张量形状，第一个值-0,会自动计算出传入的数量N（原本是N*28*28）会自动计算N

class ClassifierNet(torch.nn.Module):
    def __init__(self):
        super(ClassifierNet,self).__init__()
        self.l1 = torch.nn.Linear(784,512)  #第一层将输入维度784变成512
        self.l2 = torch.nn.Linear(512,256) #第二层
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)  #最后转换为输出维度为10 10代表图像10个分类

    #正向传播
    def forward(self,x):
        x = x.view(-1,784)  #将输入的N*0*28*28图像张量转变为N*728的张量（矩阵） 传入-1会自动计算N
        x = F.relu(self.l1(x))  #使用relu作为激活函数
        x = F.relu(self.l2(x))  #上一层的输出作为下一层的输入
        x = F.relu(self.l3(x))  #
        x = F.relu(self.l4(x))
        x = self.l5(x)  #传入SoftMax层进行分类时，不需要使用激活函数，故在第5层结束后不需要使用激活函数
        return x

model = ClassifierNet()

#2.计算损失，构建优化器
#交叉熵损失
criterion = torch.nn.CrossEntropyLoss()  #经过Softmax做完NLLLoss，这也是为什么在模型设计时没有使用Softmax

#momentum  动量、冲量  网络复杂、数据量大，使用带动量的优化算法
# 动量梯度下降就是为了解决振荡导致学习速度降低的问题。引入物理学中动量的概念，每一次梯度下降的方向和距离都受到之前梯度下降的方向和速度的影响
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#4.训练过程
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data #inputs为x_data target为y_data 图像inputs对应的第几个分类target
        outputs = model(inputs) #outputs为y_pred
        print("target ",target)
        print("target_shape ", target.shape)
        print("output ", outputs)
        print("output_shape ",outputs.shape)
        loss = criterion(outputs,target)  #计算损失

        optimizer.zero_grad()  #现将优化器中的梯度归零
        loss.backward() #反向传播
        optimizer.step() #更新梯度

        running_loss += loss.item() #当张量只有一个值时，.item()就能获取到该值，是个标量
        if batch_idx %300 == 299: #每300次训练打印一次
            print('[%d,%5d] loss: %.3f' % (epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0
#测试
def test():
    correct = 0
    total = 0
    with torch.no_grad(): #不用计算梯度
        for data in test_loader:
            x_data,y_data = data  #因为batch_size为64，故每个x_data、y_data包含64个数据组成为一个样本进行处理
            print("测试集中y_data",y_data)
            y_pred = model(x_data)  #预测
            print("y_pred",y_pred)
            #torch.max
            # torch.max(a,0)返回每一列中最大值的那个元素和该列中对应索引，返回值均为tensor，两个tensor组成一个元组
            # torchmax(a,0)返回每一行最大值的元素和该行中对应索引，各为一个tensor
            _, predicted = torch.max(y_pred.data, dim=1) #预测矩阵中每一行的最大值及对应索引
            # predicted为预测结果张量的每一行最大值的索引组成的张量，索引就对应是第几个分类
            print("测试集中predicted",predicted)
            total += y_data.size(0) #y_data是一个bacch_size*10的概率结果矩阵 .size(0)就取到第一个维度的长度N
            print('y_data.size',y_data.size(0))
            correct_t = (predicted == y_data).sum().item()  #张量perdicted和y_data中对应索引上值相等时，新的张量correct_t对应位置就是1，否则为0，
            # .sum对本张量中的值求和变成只含和的一个值的张量，再使用.item()获得该值
            correct += correct_t
    print('Accuracy on test set:%d %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()