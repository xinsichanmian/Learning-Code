from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = Planetoid(root='dataset/Cora',name='Cora')
# print(len(dataset))  # 1 Cora只有一个图
data = dataset[0]
print(data)

'''
Cora数据集：分类属于节点分类
Cora数据集包含 1个无向图，2708个节点（2708篇论文），每个节点1433个特征（每篇论文包含1433个唯一单词），共有10556/2条边，7个分类（7种论文）  
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
'''
# print(dataset[0])



class GCNnet(nn.Module):
    def __init__(self,in_chanel,out_chanel):
        super(GCNnet, self).__init__()
        self.conv1 = GCNConv(in_chanel,16)
        self.conv2 = GCNConv(16,out_chanel)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = self.conv1(x,edge_index)
        print("x_size1", x.size())
        x = F.dropout(F.relu(x))
        x = self.conv2(x,edge_index)
        x = F.log_softmax(x,dim=1)  #使用log_softmax，对应损失函数设为nll_loss
        print("x_size2",x.size())
        return x

inchanels =  dataset.num_node_features  #每个节点有1433个特征
outchanels = dataset.num_classes  #分类数 7
print("in_features=",inchanels," outchanels=",outchanels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNnet(inchanels,outchanels).to(device)
data = data.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#网络训练
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#测试
model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))