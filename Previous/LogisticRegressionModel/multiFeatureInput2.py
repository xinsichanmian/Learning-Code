from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch

#使用DataSet和DataLoader加载数据集

#Dataloader帮助加载数据
#Step1.准备数据集
#DataSet是一个抽象类，必须重新定义一个加载数据集的类来继承DataSet类
class DiabetesDataSet(Dataset):
    #初始化
    '''
    一般初始化时进行构造数据集时,当数据集整个容量不大时，可以将数据集全部读到内存中
    当数据集容量很大时，就只初始化数据集中的文件名，其中下标值也根据容量保存全部数据或者保存文件名
    '''
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)  #文本中以 , 作为数据间的分割符
        # .shape[0]获取矩阵第一维的长度（即行的数量） .shape[0]获取第二维的长度，即矩阵列的数量
        self.len = xy.shape[0]  #数据集有多少条样本
        #此处是将数据集都加载到内存保存在x_data和y_data中了
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    #实例化DiabetesDataSet的对象dataset支持下标操作dataset[index]，可以根据索引获取对应数据信息
    def __getitem__(self, index):
        #返回根据索引获得x_data和y_data对应的数据,返回值是多个时，返回结果是一个元组，即(x,y)
        return self.x_data[index], self.y_data[index]

    #可以使用 len() 获取数据集条数
    def __len__(self):
        #直接返回 构造函数__init__()中定义的len即可
        return self.len

#初始化数据集，将数据集中的数据或文件名保存在内存中
dataset = DiabetesDataSet('diabetes.csv.gz')

#使用加载器DataLoader加载出一个用于训练的数据集，一般设置四个参数
# DataSet类对象dataset，小批量操作的batch_size，shuffle是否将样本打乱再进行小批量分批成,开启线程数量num_workers
#得到的train_loader，是由每4组样本的x和y的tensor x_data、 tensor y_data构成的元组的一个DataLodaer对象（一个元素为两个tensor组成的元组的列表）
train_loader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)

#Step2.设计模型
class MultiFeatureInputModel(torch.nn.Module):
    def __init__(self):
        super(MultiFeatureInputModel,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)  #第一层，输入特征8个维度，输出特征6个维度
        self.linear2 = torch.nn.Linear(6,4)  #第二层，输入特征6个维度，输出特征4个维度
        self.linear3 = torch.nn.Linear(4,1)  #第三层，输入特征4个维度，输出特征1个维度
        self.sigmoid = torch.nn.Sigmoid()    #在这儿定义一个sigmoid函数，因为三层均需要使用激活函数，所以就当作一个属性直接用

        #注意：  如果使用的激活函数是torch.nn.ReLu，因为ReLu是将小于或等于0的输入全输出为0，所以最终输出y有可能为0
        #有可能在求损失的时候会使用logy，为了避免y为0，在前几层的计算中使用ReLu作为损失函数
        # 在forward函数最后一层计算y的值时，使用一个sigmoid函数（逻辑函数）就能避免最终输出y的值是0

    #一般我们就用一个变量x，在中间层，x就是上一层的输出y,也是下一层的输入x，所以我们就用一个x，连环使用，最后得到x就是计算出的y
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))   #计算第一层的输出结果 由输入的n*8 tensor 变成输出的n*6 tensor
        x = self.sigmoid(self.linear2(x))   #计算第二层的输出结果 由n*6 变成 n*4
        x = self.sigmoid(self.linear3(x))   #计算第三层的输出结果  由n*4 变成 n*0 实现了由特征8维输入变特征1维输出
        return x
        ''' 
        因为torch.nn.Linear类继承了torch.nn.Module类，父类里定义了__call__()函数，
        而torch.nn.Module类中的__call__()函数调用forward()函数，且返回结果就是调用forward()函数的结果
        所以可以直接用Linear类对象linear加括号传入参数linear(x),调用的就是__call__()函数，间接调用forward函数计算y_pred的值
        '''

if __name__ == '__main__':
    print(dataset[0])  #调用的是__getitem__()魔法函数，得到的是两个张量tensor的元组
    multimodel = MultiFeatureInputModel()

    # Step3. 构造损失函数和优化器
    # 构造损失函数，因为在这儿还是逻辑回归模型，只是输入维度不是1维，而是多维，所以依旧用BCEloss
    criterion = torch.nn.BCELoss(size_average=True)

    # 构造优化器
    optimizer = torch.optim.SGD(multimodel.parameters(), lr=0.01)  # 传入模型的权重参数，设置学习率lr

    # Step4. 编写训练周期
    # 设置训练周期有100个周期
    for epoch in range(0, 5):
        # 对每一个小批量进行训练 在上述中batch-size=4，即每四个样本进行一次处理，
        # 之前未用mini-batch小批量进行训练时，是将整个数据集进行一次处理
        for i, data in enumerate(train_loader, 0):
            # 0.准备数据
            inputs, labels = data  # 即 x_data,y_data = data  也可以在上一步直接写成 for i,(inputs,labels) in enumerate()
            # 1.正向传播forward
            y_pred = multimodel(inputs)  # 传入x_data
            loss = criterion(y_pred, labels)  # 通过y_pred和y计算损失
            print(epoch, i, loss.item())  # 打印周期、每个周期内小批量进行的次数，损失值
            # 2.反向传播Backward
            optimizer.zero_grad()  # 先将优化器中的梯度归零
            loss.backward()  # 再反向传播
            # 4.更新权重参数
            optimizer.step()  # 例如更新w = w- lr*grad

    x_test = torch.Tensor([[0.15, 0.83, 0.23, -0.02, 0.0, 0.31, 0.53, 0.14], [0.41, -0.21, 0.24, -0.08, 0.0, -0.21, 0.62, -0.43]])
    y_test = multimodel(x_test)
    print('测试结果', y_test.data)

'''
enumerate(sequence, [start=0]) 
传入一个可遍历的数据对象（列表，元组，字符串等）
sequence -- 一个序列、迭代器或其他支持迭代对象。
start  -- 下标起始位置

for i,e in enumerate(seq):
默认i=0，每次循环后 i+=0
'''