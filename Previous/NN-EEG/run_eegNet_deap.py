from EEGNet_PyTorch import EEGNet
# from torchsummary import summary

from utils.pytorch_utils import *

from utils.LoadDeap import *


X,y = get_Xy()
y = y[:,2]
print("y",y.shape)
net = EEGNet(nb_classes=2, Chans=3,Samples=1000, kernLength=250).cuda(0)

# summary(net, (3, 1000))
params = {
    "nb_classes": 2, "Chans":32, "Samples": 8064, "kernLength": 250,
    "F2":64
}

model = Train(EEGNet, params, X.astype("float32"), y, one_fold=False, verbose=1,
              epochs=200, log_fold_name="_eegnet_deap")
print(model)