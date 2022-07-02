import torch

batch_size = 2
seq_len = 3  #定义序列 例如输入包含连续三个输入x，x1,x2,x3
input_size = 4 #输入维度
hidden_size = 2 #隐藏维度

#定义RNNCell
rnncell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

#输入数据集  格式上 张量(seq_len,batch_size,input_size) 三维张量
dataset = torch.randn(seq_len,batch_size,input_size)  #定义数据集输入rand和randn的第一个参数是可变参数size，自定义张量形状
print("dataset",dataset)
#h0输入一个都为0的向量
hidden = torch.zeros(batch_size,hidden_size)
print("hidden",hidden)
print("hidden.size",hidden.size())
print("-------------------")

#循环神经网络计算
for idx,input in enumerate(dataset,0):
    print('-'*20,idx,"-"*20)
    print('Input_size',input.size())  #张量.size()与张量.shape相同，.size(0)与.shape[0]一样
    #上一层rnncell输出hidden作为下一层rnncell输入hidden
    #输入rnncell的输入向量input 二维张量(batch_size,input_size)
    hidden = rnncell(input,hidden)  #计算hidden
    print("hidden_size",hidden.shape)  #(batch_size,hidden_size)
    print(hidden)


#b,1024,1,1    1,b,1024,