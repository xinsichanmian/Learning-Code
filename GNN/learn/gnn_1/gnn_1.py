import numpy as np
import pandas as pd
import networkx as nx

# dic={"clm0":{"idx0":1,"idx1":2,"idx2":3},
#      "clm1":{"idx0":4,"idx1":5,"idx2":6},
#      "clm2":{"idx0":7,"idx1":8,"idx2":9}}
# df=pd.DataFrame(dic)
# print(df)

edges = pd.DataFrame()
edges['sources'] = [1,1,1,2,2,3,3,4,4,5,5,5]  #源节点
edges['targets'] = [2,4,5,3,1,2,5,1,5,1,3,4]  #目标节点  如节点1与节点2、4、5相连
edges['weights'] = [1,1,1,1,1,1,1,1,1,1,1,1]  #边的权重

G = nx.from_pandas_edgelist(edges,source='sources',target='targets',edge_attr='weights')

#节点的度degree
print("节点的度")
print(nx.degree(G))

#连通分量
print("连通分量")
print(list(nx.connected_components(G)))

#图直径  图上两两节点连接的最长距离
print("图直径")
print(nx.diameter(G))

#度中心性
print("度中心性")
print(nx.degree_centrality(G))

#特征向量中心性
print("特征向量中心性")
print(nx.eigenvector_centrality(G))

#betweenness
print("betweenness")
print(nx.betweenness_centrality(G))

#clossness
print("clossness")
print(nx.closeness_centrality(G))

#pagerank
print("pagerank算法")
print(nx.pagerank(G))

#HITS
print(("HITS算法"))
print(nx.hits(G))