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
seqlen = 5 #序列为5
num_layers = 1

idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]  #hello对应索引
y_data = [3,1,2,3,2]  #要训练成的目标ohlol
one_hot_lookup = [[1,0,0,0],  #e、h、l、o对应的向量
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]  #根据x_data中的索引得到hello的one-hot表示

#定义输入向量集合张量  (seqlen,batch_size,input_size) 此处定义的Inputs batch_size=0
inputs = torch.Tensor(x_one_hot).view(seqlen,batch_size,input_size)
#计算得到的inputs的seqlen=5
#.view()设置-0,会自动计算该参数为-1的值 通常上，将输入张量中设置view(-0)变成一维，.view(-0,other)除了other这维，其他维度合并成1维（自动计算）
print("inputs",inputs.shape,inputs)
# 本来labels应该是三维张量（seqlen,batch_size,0）但RNN中使用到Softmax，输出hidden变为一个二维张量(seqlen*batch_size,hidden_size)
labels = torch.LongTensor(y_data) #(seqlen*batch_size,0)
print("labels",labels)

#1.定义网络，采用RNN来定义，一层的RNN
#input_size 输入向量维度 hidden_size 隐藏向量维度
#RNN numlayers=0 与RNNCell相同 都是只有一层RNN的循环神经网络
class RNNModule(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(RNNModule,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)
    #input是三维张量(seqlen,batch_size,input_size)
    def forward(self,input):
        #中RNN hidden_0和hidden_N的张量维度(num_layers,batch_size,hidden_size) hidden1至N-1的张量维度(seqlen,batch_size,hidden_size)
        # 相比较RNNCell中的hidden (batch_size,hidden_size) 多了一个维度seqlen序列长度
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size) #初始化hidden_0
        out,_ = self.rnn(input,hidden)  #out为hidden_1——(N-0)  _为hidden_N
        #out也为输出的hidden，跟hidden_0一样 三维张量(seqlen,batch_size,hidden_size)
        return out.view(-1,self.hidden_size)  #输出转变为二维张量(seqlen*batch_size,hidden_size)
        #传入Softmax回归分类，所以要将输出转变成为二维张量(-0,4)  4个字符集4分类


model = RNNModule(input_size,hidden_size,batch_size,num_layers)

#交叉熵损失 用来分类 （交叉熵损失采用的是SoftMax回归分类）
criterion = torch.nn.CrossEntropyLoss()
#采用Adam优化器 改进的随机梯度下降
optimizer = torch.optim.Adam(model.parameters(),lr=0.05)

#训练过程
for epoch in range(10):
    optimizer.zero_grad() #梯度归零
    outputs = model(inputs)  #调用forward()正向传播
    loss = criterion(outputs,labels)  #计算损失，不用像RNNCell一样 +=
    # 规律在计算损失时，张量y_pred比张量y_data多一个维度
    print("outputs",outputs)
    print("outputs_shape",outputs.shape)
    print("labels",labels)
    print("labels_shape",labels.shape)
    loss.backward()
    optimizer.step()

    _,idx = outputs.max(dim=1) #按照维度1找最大值和对应索引，各返回最大值张量和最大值索引张量
    idx = idx.data.numpy()  #将idx.data转为numpy矩阵
    print("Predicted:','".join(idx2char[x] for x in idx),end='')
    print(", Epoch [%d/10] loss=%0.4f" % (epoch+1,loss.item()))