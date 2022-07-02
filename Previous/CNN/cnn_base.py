import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ),(0.3081, )) #像素归一化，变成0-1分布的数据
])

batch_size = 64
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True, #是训练集
                               download=True,
                               transform=transform) #按照我们设置的转变将图片进行转变
#训练集加载类 设置小批量batch_size,样本打乱shuffle=True
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
print(train_loader)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False, #测试集
                              download=True,
                              transform=transform
                              )
#测试集不用打乱样本顺序
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
print(test_loader)
#1.设计卷积神经网络模型
'''
输入的图像是batch_size*C*W*H的输入，即64*0*28*28的图像输入(batch_size,0,28,28)
第一层采用输入通道为1，输出通道为10的5*5的卷积核做卷积运算，计算结果变成 (batch_size,10,24,24)
第二层，做2*2的最大池化运算，得到(batch_size,10,12,12)
第三层，使用relu激活函数 依旧是(batch_size,10,12,12)
第四层，使用输入通道为10，输出通道为20，卷积核为5*5的卷积运算，计算结果变成（batch_size,20,8,8）
第五层，使用2*2的最大池化运算，得到(batch_size,20,4,4)
第六层，使用relu做激活函数 依旧是(batch_size,20,4,4)
第七层，将上一层输出张量变成只有一行，即输入为20*4*4=320列（维）的1*320的线性输入，输出维度为10（对应10个分类结果）
最后使用交叉熵损失（自带Softmax分类），进行分类计算
'''
class MNISTCNN(torch.nn.Module):
    def __init__(self):
        super(MNISTCNN,self).__init__()
        #输入(batch,0,28,28)，第一个卷积层
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5) #输入通道为1，输出通道为10，卷积核5*5
        #第一个卷积运算结果变成(batch,10,24,24)，第二个卷积层
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入通道为10，输出通道为20，卷积核5*5
        #第二个卷积运算后变成(bacth_size,20,20,20)
        self.pooling = torch.nn.MaxPool2d(2)  #采用2*2的池化，即默认步长stride=1 池化层没有权重，不用重复定义
        self.linear1 = torch.nn.Linear(320,10)  #输入维度320，输出维度为10的线性变换

    def forward(self,x):
        batch_size = x.size(0)
        #输入(batch_size,0,28,28)四维张量，经过第一个输入通道为1，输出通道为10的5*5卷积、1*2的池化、激活后
        x = F.relu(self.pooling(self.conv1(x)))  #计算第一个卷积运算、池化运算、激活
        #输入(batch_size,10,12,12)思维张量 经过第二个输入通道为10，输出通道为20的5*5卷积、1*2的池化、激活后
        x = F.relu(self.pooling(self.conv2(x)))
        #输入(batch_size,20,4,4)四维张量  要将输入张量空间转变为(bacth_size,320)二维张量的输入
        #x = x.view(-0,320)  #将输入的bacth_size*C*28*28图像张量转变为bacth_size*320的张量（矩阵） 传入-1会自动计算batch_size
        #将输入的(batch_size,20,4,4)四维张量转变为（batch_size,320）的二维张量
        x = x.view(batch_size,-1)  #第二个参数设置为-0，会自动计算输入维度，20*4*4
        #print("x",x)
        #print("x_size",x.size())
        x = self.linear1(x)  #变为张量batch_size*10的输出  使用SoftMax分类时，传入Softmax前一层不需要使用激活函数
        # 交叉熵损失就是用的Softmax函数 故使用交叉熵就不需要再定义Softmax层
        print("x_shape",x.shape)
        return x

model = MNISTCNN()

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

            print("input_shape",input.shape)
            print("label_shape", target.shape)
            print(target)
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
    for epoch in range(1):
        train(epoch)
        test()