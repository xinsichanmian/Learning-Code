from torch_geometric.datasets import  Planetoid
from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = Planetoid(root='dataset/Cora',name='Cora')
# print(len(dataset))  # 1 Cora只有一个图
data = dataset[0]
print(data)

class GATnet(nn.Module):
    def __init__(self,in_features,out_features):
        super(GATnet, self).__init__()
        # heads多头注意力,默认为1 下一层gatconv的in_channels=上一层的out_channels*上一层的heads  contat：如果设置为False，则多头注意力被平均而不是连接。（默认值：True）
        self.gatconv1 = GATConv(in_channels=in_features,out_channels=8,heads=8,dropout=0.6)
        self.gatconv2 = GATConv(in_channels=8*8,out_channels=out_features,heads=1,concat=False,dropout=0.6)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = F.elu(self.gatconv1(x,edge_index))
        x = F.dropout(x,p=0.6,training=self.training)
        x = self.gatconv2(x,edge_index)
        # print("x_size1",x.size())
        # print(x)
        x = F.log_softmax(x,dim=-1)
        # print("x_size2",x.size())
        # print(x)
        return x

inchanels =  dataset.num_node_features  #每个节点有1433个特征
outchanels = dataset.num_classes  #分类数 7
print("in_features=",inchanels," outchanels=",outchanels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATnet(inchanels,outchanels).to(device)
data = data.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

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