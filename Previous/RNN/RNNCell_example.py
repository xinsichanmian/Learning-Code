import torch

#训练一个模型 hello ——>ohlol

#将hello采用One-Hot Vectors 杜热向量 表示
'''
<字母,索引> <e,0> <h,0> <l,1> <o,2> 
h 0 ——> 0 0 0 0 ——> x1
e 0 ——> 0 0 0 0 ——> x2 
l 1 ——> 0 0 0 0 ——> x3
l 1 ——> 0 0 0 0 ——> x4
o 2 ——> 0 0 0 0 ——> x5   变成一个序列为5的向量输入 batch_size=0

则hello的One-Hot向量转换后 input_size=4
'''
input_size = 4  #hello中总共有4个不一样的字符，采用one-hot向量表示，就有4维，故input_size=4
batch_size = 1  #
hidden_size = 4  #隐藏向量的维度为4 代表4个字符的输出，再在hidden后面接一个Softmax回归用来分类即可得到对应概率，概率最大的索引值就对应字符的预测


idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]  #hello对应索引
y_data = [3,1,2,3,2]  #要训练成的目标ohlol
one_hot_lookup = [[1,0,0,0],  #e、h、l、o对应的向量
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]  #根据x_data中的索引得到hello的one-hot表示

#定义输入向量集合张量  (seqlen,batch_size,input_size) 此处定义的Inputs batch_size=0
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
#.view()设置-0,会自动计算该参数为-1的值 通常上，将输入张量中设置view(-0)变成一维，.view(-0,other)除了other这维，其他维度合并成1维（自动计算）
print("inputs",inputs.shape,inputs)
#目标结果集合 张量
labels = torch.LongTensor(y_data).view(-1,1)  # (seqlen,0) 即(5,0)
print("labels",labels)

#1.定义网络，采用RNNCell来定义，一层的RNN
#input_size 输入向量维度 hidden_size 隐藏向量维度

class RNNCellModule(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(RNNCellModule,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size) #定义rnncell

    def forward(self,input,hidden):
        hidden = self.rnncell(input,hidden)  #传入输入向量和hidden_0计算hidden
        return hidden

    #初始化hidden_0
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

model = RNNCellModule(input_size,hidden_size,batch_size)

#交叉熵损失 用来分类 （交叉熵损失采用的是SoftMax回归分类）
criterion = torch.nn.CrossEntropyLoss()
#采用Adam优化器 改进的随机梯度下降
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

#训练过程
for epoch in range(10):
    loss = 0
    optimizer.zero_grad()  #每轮循环训练开启之前先让梯度归零
    hidden = model.init_hidden()  #初始化hidden_0
    print('Predicted string ',end='')
    for input,label in zip(inputs,labels):
        print("----------------")
        print("input---",input)
        print("label--",label)  #labels是二维张量(seqlen,0) label就是一维张量(0)
        hidden = model(input,hidden)  #计算隐藏向量 调用forward()函数 输入的input 二维张量(batch_size,input_size)
        # hidden 二维张量(bach_size,hidden_size)
        loss += criterion(hidden,label)  #计算损失，因为有多个隐藏向量损失，需要做计算图来计算损失
        #规律在计算损失时，张量y_pred比张量y_data多一个维度
        _,idx = hidden.max(dim=1) #隐藏  hidden 二维张量(bach_size,hidden_size)

        print("hidden--",hidden)
        print(idx2char[idx.item()],end='')
    loss.backward()  #反向传播
    optimizer.step()  #更新梯度
    print(", Epoch [%d/10] loss=%0.4f" % (epoch+1,loss.item()))