import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

'''
  使用pyG进行图神经网络操作时，输入的图数据要符合torch_geometric.data的Data格式，即如下的格式
'''

# 边  shape = [2,num_edge]  节点连接采用COO格式
edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]], dtype=torch.long)

# 两种方法表示边都可以  此处在调用的时候先转置.t()再加上continguous()方法
edge_index2 = torch.tensor([[0, 1],
                            [1, 0],
                            [2, 1],
                            [0, 3],
                            [3, 2]], dtype=torch.long)

# 节点  shape = [num_nodes,num_node_features]
x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)  # 4个节点，每个节点两个特征值

# 节点真实标签
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

data1 = Data(x=x, y=y, edge_index=edge_index)
#
data2 = Data(x=x, y=y, edge_index=edge_index2.t().contiguous())  # 转置.t()再加上continguous()方法
#
print(data1)  # Data(x=[4, 2], edge_index=[2, 4], y=[4])
print(data2)
# import torch
# from torch_geometric.data import Data
#
# #边，shape = [2,num_edge]
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# #点，shape = [num_nodes, num_node_features]
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index)
