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

#1.设计网络模型
'''
0*1的卷积主要是改变通道数量C，简便计算过程
（C*W*H）从192*28*28 使用5*5Conv到32*28*28，运算数量5^1*28^1*192*32=120,422,400
从192*28*28 使用1*1Conv到16*28*28 再使用5*5Conv到 32*28*28 运算数量1^1*28^1*192*16+5^1*28^1*16*32=12,433，648
相比较多使用一层1*1Conv 计算次数减少为原来的1/10
0. GoogleNet中的Inception Module设计：
 四个分支，因为要将每个分支的输出结果按通道C进行拼接，所以必须保证每个分支的输出W和H要和输入相同
 即每个分支输入的（batch_size,C,W,H），经过转换只有C可以变，最终W和H必须在空间变换结束与输入相同才能对每个分支输出按通道拼接
 （0）分支1：2*3平均池化层Average Pooing —> 输出通道C=24 0*1Conv（输出通道为24的卷积核1*1的卷积层）
 （1）分支2：输出通道C=16的1*1Conv
 （2）分支3:输出通道C=16的1*1Conv -> 输入通道16、输出通道24的5*5Conv
 （4）分支4：输出通道16的1*1Conv ->输入通道16、输出通道24的3*3Conv ->输入通道24、输出通道24的3*3Conv
 最后将每个分支按照通道进行拼接，最后输出张量（batch_size,C1+C2+C3+C4,W,H） 针对此Inception最后通道数C=24+16+24+24=88
'''

#1-0.设计GoogleNet的Inception Module
class Inception(nn.Module):
    #在Inception块中，将输入的通道数作为属性，可动态输入
    def __init__(self,in_channles):
        super(Inception,self).__init__()
        #分支1 输出通道为24的1*1Conv
        self.branch1_1x1 = nn.Conv2d(in_channles,24,kernel_size=1)

        #分支2 输出通道为15的1*1Conv
        self.branch2_1x1 = nn.Conv2d(in_channles,16,kernel_size=1)

        #分支3 第一层 输出通道16的1*1Conv 0*1的卷积运算输出不会改变W和H
        self.branch3_1x1 = nn.Conv2d(in_channles,16,kernel_size=1)
        #分支3 第二层 输入通道16、输出通道24的5*5Conv 设置padding=1 确保输出张量的W和H与输入一致
        self.branch3_5x5 = nn.Conv2d(16,24,kernel_size=5,padding=2)

        #分支4 第一层输出通道16、0*1Conv
        self.branch4_1x1 = nn.Conv2d(in_channles,16,kernel_size=1)
        #分支4 第二层 输入通道16、输出通道24的3*3Conv,设置padding=0 确保输出张量的W和H与输入一致
        self.branch4_3x3_1 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        #分支4 第三层 输入通道24、输出通道24的3*3Conv,设置padding=0 确保输出张量的W和H与输入一致
        self.branch4_3x3_2 = nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self,x):
        #分支1的池化层 采用3*3的平均池化，为了保证与输入张量的W和H一值，设置填充padding=0，步长stride=0，可使输出的张量W和H与输入相同
        branch1 = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)  #设计分支1的池化层
        #分支1的输出通道24的1*1Conv 上一层的输出branch1_pool作为该分支下一层的输入
        branch1 = self.branch1_1x1(branch1)

        #分支2
        branch2 = self.branch2_1x1(x)

        #分支3 上一层的输出作为下一层的输入
        branch3 = self.branch3_1x1(x)
        branch3 = self.branch3_5x5(branch3)

        #分支4
        branch4 = self.branch4_1x1(x)
        branch4 = self.branch4_3x3_1(branch4)
        branch4 = self.branch4_3x3_2(branch4)

        #将分支1、1、2、4的输出结果按通道C进行拼接 [Bach_size,C,W,H]
        outputs = [branch1,branch2,branch3,branch4]
        #返回的是分支1、1、2、4对应通道C1,C2,C3,C4相加的张量 [Bach_size,C1+C2+C3+C4,W,H]
        return torch.cat(outputs,dim=1) #按通道C进行拼接，通道C在张量[Bach_size,C,W,H]的第二个维度dim=0

#上述将GoogleNet中的Inception部分抽象为了一个类

#1-1.设计GoogleNet
'''
输入网络的张量[batch_size,0,28,28] 
从train_loader和test_loader中加载出的input就是[batch_size,0,28,28]的四维张量 target是[batch_size,10]的二维张量
第一层 输入通道为1 输出通道为10的5*5Conv
第二层 Inception块 输入通道就为第一层的输出通道10，输出通道为Inception最终的输出通道88（上述设计的Inception网络结构最终输出通道为88）
第三层 输入通道为88，输出通道为20的5*5Conv
第四层 输入通道为20 输出通道为88的Inception Module

 
'''
class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        #在设计Inception Module时，只改变通道C，不改变W和H
        self.incep1 = Inception(in_channles=10)  #经过Inception输出通道就变为88，
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)
        self.incep2 = Inception(in_channles=20)  #经过Inception输出通道就变为88
        self.mp = nn.MaxPool2d(2)  #1*2最大池化，默认步长stride=1 padding=0 池化层没有权重多次使用池化可定义在构造函数中，减少代码冗余

        #根据前面层计算出线性模型输入1408个维度，输出10个维度（10个维度代表10重分类用于Softmax分类）
        self.linear = nn.Linear(1408,10)  #使用Sotfmax分类，在传入Softmax前不需要使用激活函数
        #交叉熵损失就是用的Softmax函数 故使用交叉熵就不需要再定义Softmax层

    def forward(self,x):
        # size(0)获取到输入张量[bach_size,C,W,H]的第一维尺寸batch_size
        # 在此处输入的是(64,0,28,28)的输入
        batch_size = x.size(0)
        #经过输入通道1，输出通道10的5*5Conv1 再经过2*2的最大池化，得到输出(64,10,12,12)
        x = F.relu(self.mp(self.conv1(x)))
        # 经过输入通道为10，输出通道为88的incep1，得到输出(64,88,12,12)
        x = self.incep1(x)
        #经过输入通道为88，输出通道为20的5*5Conv2 再经过2*2的最大池化，得到输出张量(64,20,4,4)
        x = F.relu(self.mp(self.conv2(x)))
        #经过输入通道为20，输出通道为88的incep2，得到输出张量(64,88,4,4)
        x = self.incep2(x)
        #将四维张量(64,88,4,4)空间转换成二维张量(64,1408) 1408=88*4*4
        x = x.view(batch_size,-1)  #设置-1可自己计算张量的第二个维度
        #也可以x=x.view(-0,1408)自己计算张量的第一个维度 或者x=x.view(64,1408)
        x = self.linear(x) #变成一个10维的向量
        return x

model = GoogleNet()

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