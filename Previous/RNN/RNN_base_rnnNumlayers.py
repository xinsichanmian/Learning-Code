import torch

batch_size = 32
seq_len = 3  #定义序列 例如输入包含连续三个输入x，x1,x2,x3
input_size = 64 #输入维度
hidden_size = 32 #隐藏维度
num_layers = 2  #整个RNN中有几层RNN

'''
num_layers = 1  #RNN的层数，定义层RNN，RNN的计算是很耗时的
cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

out,hidden_N = cell(inputs,hidden_0)
#out为所有RNNCell输出的结果集合hidden1至(N-0)  是个三维张量（seqlen,batch_size,hidden_size）
#hidden_N 为整个RNN的输出结果  hidden是个三维张量(num_layers,batch_size,hidden_size)
#inputs为输入序列集合 是个三维张量(seqlen,batch_size,input_size)
#hidden_0为输入的hidden0，即输入RNN的第一个hidden
#上述seqlen表示输入向量x的序列长度 如x1,x2,x3作为一个序列输入 
input_size为输入向量x的维度,hidden_size为隐藏向量的维度
num_layers为该RNN中有几层RNN，batch_size小批量
'''


cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

inputs = torch.randn(seq_len,batch_size,input_size)  #输入向量 是三维张量
print(inputs.shape,inputs)
hidden_0 = torch.zeros(num_layers,batch_size,hidden_size) #整个网络的RNN的每一层的h_0
print("hidden_0",hidden_0)

#将hidden0传入RNN中，输出out为hidden_1——(N-0)  hidden为hidden_N
out,hidden = cell(inputs,hidden_0)
#out的维度维与输入维度一致都是(seq_len,batch_size,input_size)
#hidden_N的维度与hidden_0一致都是(num_layers,batch_size,hidden_size)
#hidden  (seq_len,batch_size,hidden_size)
#

print("------------")
print("Output_size",out.shape)
#print("Output",out)
print("Hidden_size",hidden.shape)
#print("Hidden",hidden)

'''
torch.nn.RNN()  一些参数
初始输入向量数据集 三维张量(seqlen,batch_size,input_size)
batch_first：若为True，就将batch_size放到输入向量的第一个位置，即输入数据集就为(batch_size,seqlen,input_size)

'''
