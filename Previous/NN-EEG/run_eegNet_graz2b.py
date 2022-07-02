from EEGNet_PyTorch import EEGNet
# from torchsummary import summary
from utils.pytorch_utils import *
from utils.LoadGraz2B import *
net = EEGNet(nb_classes=2, Chans=3,Samples=1000, kernLength=250).cuda(0)
# summary(net, (3, 1000))
X,y=get_Xy2(dataset_dir = "../dataset/BCICIV_2b_grazdata")
y=y.astype("int")
print(y)
params = {
    "nb_classes": 2, "Chans":3, "Samples": 1000, "kernLength": 250,
}

model = Train(EEGNet, params, X.astype("float32"), y, one_fold=False, verbose=1,
              epochs=200, log_fold_name="graz2b_01")
print(model)