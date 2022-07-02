import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ),(0.3081, ))
])

batch_size = 64
#0.读取数据
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True, #是训练集
                               download=True,
                               transform=transform) #按照我们设置的转变将图片进行转变
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False, #非训练集，即测试集
                              download=True,
                              transform=transform
                              )
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


#ResidualNet（残差网络） 主要解决梯度消失的问题
'''
Resuidual Block
跳过两个层，第一层输入x,第二层输出f(x) 第三层输入Z=f(x)+x 要保证C、W、H都相同才能相加
注：在第三层输入前，做完加法再做激活
'''
#1-0.定义残差块
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        #残差块的第一层，使用输入通道和输出通道均相同，2*3Conv，为了保证输出张量的C、W、H与输入相同，设置padding=1即可
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        #残差块第二层，使用3*3Conv，同理为了为了保证输出张量的C、W、H与输入相同，设置padding=1即可
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        z = F.relu(y+x)  #第三层输入z，先执行第一层输入x和第二层输出y的加法运算再做激活，
        # 残差块的输出中使用了激活函数，故在残差网络设计时，不需要再对残差块的计算结果再进行激活
        return z

#1-1.设计残差网络
'''
对MNIST数据集分类 使用Dataloader加载数据，输入的是(bacth_size,0,28,28)四维张量
第1-3层：输入通道为1、输出通道16的5*5Conv 使用relu做激活函数,再使用2*2的最大池化（默认stride=1）
第4层：残差块，经过该层不改变通道C、宽度W和高度H 输出结果为(batch_size,16,12,12)
第5-7层：输入通道16，输出通道32的5*5Conv，使用relu做激活函数，再使用2*2的最大池化（默认stride=1）
第8层：残差块，经过该层，输出结果为(batch_size,32,4,4)
第9层：线性模型，将输入的四维张量(batch_size,32,4,4)转变为(batch_size,10)的输出张量，
      先要对四维张量进行空间转变为二维（batch_size,512），即输入维度512，输出维度10，再使用Softmax进行分类
      交叉熵损失包含了Softmax层，故在设计网络模型时，无需单独设计Softmax层，直接使用交叉熵损失作为损失函数即可
'''
#一个规律：在构造函数__init__()中使用torch.nn.Sgmoid或torch.nn.AvgPool2d等，即使用torch.nn.层 来作为一层
#而在forward()函数，使用torch.nn.functional as F中使用构造函数未定义的F.sgmoid、F.avg_pool2d来计算
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()

        self.conv1 = nn.Conv2d(1,16,kernel_size=5)
        self.rblock1 = ResidualBlock(16) #第一个残差块,残差块不改变C、H、W
        self.mp = nn.MaxPool2d(2) #池化层，第一个参数是kernel_size，为2，stride默认值是kernel_size
        #池化层没有权重，设计网络时，一样的池化层定义一个即可
        self.conv2 = nn.Conv2d(16,32,kernel_size=5)
        self.rblock2 = ResidualBlock(32) #承接上一层输出通道，残差块的输入通道32
        self.linear = nn.Linear(512,10)  #32*4*4=512作为输入维度，10作为分类数量作为输出10维度，再送入Softmax回归分类

    def forward(self,x):
        # 输入四维张量(batch_size,0,28,28)
        batch_size = x.size(0)  #.size(0)获取到batch_size
        x = self.mp(F.relu(self.conv1(x))) #输出张量(batch_size,16,12,12)
        x = self.rblock1(x)  #在设计的残差块有计算x+y的激活，故在此处不需要再做激活
        x = self.mp(F.relu(self.conv2(x))) #输出张量(batch_size,32,4,4)
        x = self.rblock2(x)
        x = x.view(batch_size,-1) #将四维张量(batch_size,32,4,4)转变为二维张量(batch_size,512)
        x = self.linear(x)
        return x

model = ResidualNet()

#将模型迁移到GPU上进行运算  如果cuda可使用，使用cuda:0 0代表第一块显卡，不存在显卡就使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#将模型放到GPU(cuda)  后面的训练和测试都要将迁移到同一块显卡上
model.to(device)

#2.构建损失函数和优化器
#交叉熵损失
criterion = torch.nn.CrossEntropyLoss() #经过Softmax做完NLLLoss，这也是为什么在模型设计时没有使用Softmax
#优化器 传入模型参数，设置学习率，使用动量梯度下降
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#4.训练过程
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        input,target = data

        #对应前面模型，也要将训练集运算迁移到同一块显卡上（GPU）
        input,target = input.to(device),target.to(device)

        output = model(input) #预测结果
        loss = criterion(output,target)  #计算损失
        # 规律在计算损失时，张量y_pred比张量y_data多一个维度

        optimizer.zero_grad() #梯度归零后再进行梯度更新
        loss.backward() #反向传播 计算梯度
        optimizer.step()  #更新梯度
        running_loss += loss.item()  #求损失
        if(batch_idx%300 == 299):  #每300循环打印一次训练结果
            print('[%d,%5d] loss: %.3f' % (epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():  #不计算权重
        for idx,data in enumerate(test_loader,0):  #每个batch进行一次操作
            input,target = data  # target是包含有batch_size个结果的张量

            #对应前面模型，也要将测试集运算迁移到同一块显卡上（GPU）
            input,target = input.to(device),target.to(device)

            output = model(input)  #预测结果
            _,predicted = torch.max(output.data,dim=1)  #将output.data矩阵中每一行最大值的下标取出来对应的就是预测的数字分类
            #predicted是output张量每一行最大值对应索引值组成的张量
            total += output.size(0)  #output是batch_size*10的概率结果矩阵，.size(0)获取到batch_size
            correct_t = (target==predicted).sum().item()  #将target和predicted张量对应位置值相等时,在correct张量对应位置就是1，否则就为0
            correct += correct_t
    print('Accuracy on test set:%d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(4):
        train(epoch)
        test()