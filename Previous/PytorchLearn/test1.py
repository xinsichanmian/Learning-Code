import torch
import numpy as np
import mne

a1 = np.array([1.,0.,1.,0.,1.,1.,0.])
a2 = np.array([1.,1.,1.,0.,1.,1.,0.])
a3 = np.array(a1==a2).astype(int).sum()
print(a3)
# a1 = [[[1, 2, 3],
#        [1, 2, 3]],
#       [[1, 1, 1],
#        [1, 1, 1]],
#       [[2, 2, 2],
#        [2, 2, 2]]]
# a2 = [[[1, 1, 1],
#        [1, 1, 1]],
#       [[2, 2, 2],
#        [2, 2, 2]],
#       [[3, 3, 3],
#        [3, 3, 3]]]
# ar_a1 = np.array(a1)
# ar_a2 = np.array(a2)
# ar_a3 = np.vstack((ar_a1,ar_a2))
# # print(ar_a1.shape)
# # print(ar_a3.shape)
# # print(ar_a3)
# l1 = np.array([1,2,2])
# l1 = np.concatenate((l1,np.array([2,3,1])))
# print(l1)

# ar = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0.]
# t_a = torch.Tensor(ar)
# print(t_a)

# path = "dataset/BCICIV_2b_gdf/"
# filename = "B0101T"
# event_type2idx = {276: 0, 277: 1, 768: 2, 769: 3, 770: 4, 781: 5, 783: 6, 1023: 7, 1077: 8, 1078: 9, 1079: 10,
#                   1081: 11, 32766: 12}
# raw = mne.io.read_raw_gdf(path + filename + ".gdf",preload=True, stim_channel=-1)
# data = raw._data
# #print(raw._data)
# data1 = data[0]
# print("length",raw._raw_extras[0]['events'][0])
# print("position",raw._raw_extras[0]['events'][1])
# print("event type",raw._raw_extras[0]['events'][2])
# print("event index",[event_type2idx[event_type] for event_type in raw._raw_extras[0]['events'][2]])
# print("duration",raw._raw_extras[0]['events'][4])
# print("CHN",raw._raw_extras[0]['events'][3])

# print(raw._raw_extras[0]['ch_names'])
# print(raw.info['sfreq'])
# print("length",raw._raw_extras[0]['events'][0])

# tensor1 = torch.Tensor([[1,2,3,4],   #2*4的张量，
#                        [3,2,5,1],
#                        [2,4,2,1]])
#
# maxValue,idx = torch.max(tensor1,dim=1)
# print(idx.data,tensor1.size(1))
# t2 =  tensor1.sort(dim=0,descending=True)
# print(t2)
# tensor2 = torch.randn(2,3)
# print("tensor2",tensor2)
# arrary = np.array([[1,2,3]])
# print(arrary.shape)

# X_train = np.random.rand(10, 2, 6, 4).astype('float32')
# print(X_train)

# n1 = np.array([[1,2,3,4],
#       [2,3,4,5]])
# n2 = np.array([[0,2,3,4],
#       [2,3,4,5]])
# print((n1==n2).sum())
