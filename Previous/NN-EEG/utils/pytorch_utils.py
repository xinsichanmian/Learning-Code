from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # PyTorch v0.4.0
torch.seed()

# device = torch.device("cpu")


from functools import reduce
from operator import __add__

def SamePadding( kernel_size = (4, 1)):
# Internal parameters used to reproduce Tensorflow "Same" padding.
# For some reasons, padding dimensions are reversed wrt kernel sizes,
# first comes width then height in the 2D case.
    conv_padding = reduce(__add__, 
        [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    
    pad = nn.ZeroPad2d(conv_padding)
    return pad

# kernel_sizes = (4,3)
# conv = nn.Conv2d(1, 1, kernel_size=kernel_sizes)
# pad = SamePadding(kernel_size=kernel_sizes)

# x = torch.randn(size=(1, 1, 103, 40))
# print(x.shape) # (1, 1, 103, 40)
# print(conv(x).shape) # (1, 1, 100, 40)
# print(conv(pad(x)).shape) # (1, 1, 103, 40)

def Predict(model, X):
    batch_size = 512
    res = np.array([])

    with torch.no_grad():
        model.eval()
        for i in range(0, len(X), batch_size):
            s, e = i, i + batch_size

            inputs = torch.from_numpy(X[s:e]).to(device)
            pred = model(inputs)
            # res.append(pred.cpu().numpy().argmax(axis=1))
            res = np.concatenate([res, pred.cpu().numpy().argmax(axis=1)])
        model.train()
    return np.array(res).flatten()


def Evaluate(model, X, Y, params=["acc"]):
    """
    function的功能：评估nn模型。

    Parameters
    ----------
    model:torch.nn
        模型
    X: ndarray, type=float32
        数据
    y: ndarray, type=int64
        标签
    params: str
        ["acc","auc", "recall","precision","fmeasure"]
    """
    results = []

    predicted = Predict(model=model, X=X)
    # print(type(Y))
    # print(Y.shape)
    
    # print(type(predicted))
    # print(predicted.shape)
     
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall / (precision+recall))
    return results


def Fit(model, X_train, y_train, validation_data,  criterion, epochs=120, batch_size=300, lr=0.001, verbose=0, log_fold_name="logs"):
    """
    function的功能：nn模型拟合。

    Parameters
    ----------
    X_train: ndarray, type=float32
        训练数据
    y_train: ndarray, type=int64
        训练标签
    validation_data: list, ndarray
        测试用的数据与标签
    criterion: torch.nn.loss,
        loss Fun
        nn.CrossEntropyLoss() 、nn.MSELoss() ……
    verbose: int,
        if 0, no verbose
        if 1, print training log
    ...

    Examples
    --------
    >>> fit(X_train, y_train, validation_data = [X_test,y_test], criterion=nn.CrossEntropyLoss())
    """
    with SummaryWriter(comment=log_fold_name) as writer:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # for epoch in tqdm(range(epochs)):
        for epoch in range(epochs):
            correct = 0
            total = 0
            train_size = X_train.shape[0]
            for i in range(0, train_size, batch_size):
                s, e = i, i + batch_size

                X_batch = torch.from_numpy(X_train[s:e]).to(device)
                y_batch = torch.from_numpy(y_train[s:e]).to(device)

                outputs = model(X_batch)

                loss = criterion(outputs, y_batch)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # computy accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += len(y_batch)
                correct += (predicted == y_batch).sum().item()

                niter = i // batch_size + epoch * (train_size // batch_size)
                writer.add_scalar('Train/Loss', loss.item(), niter)

            correct_trn = 100 * correct / total
            writer.add_scalar('Train/EpochAcc', correct_trn, epoch)

            # 开始测试
            if validation_data:
                X_test, y_test = validation_data
                
                dev_acc = Evaluate(model, X_test, y_test)[0]
                writer.add_scalar('Test/EpochAcc', dev_acc, epoch)

            if verbose and epoch % 10 == 0:
                # log 日志
                log_str = f'epoch[{epoch}/{epochs}] --- Train loss:%.4f, Train accuracy:%.2f%%. ' % (loss.item(), correct_trn)
                if validation_data:
                    log_str += "Dev accuracy:%.2f.%%" % ( dev_acc )
                print(log_str)
    
    if validation_data:
        return dev_acc        
    else:
        return None


def Train(ModelClass, params, X, y, validation_data=None, n_splits=5, one_fold=True, lr=0.001, epochs=90, batch_size=64, verbose=0, log_fold_name="logs"):
    # 5-fold cross-validation
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    dev_score = []
    test_score = []

    criterion = nn.CrossEntropyLoss()
    for k, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_dev, y_dev = X[test_index], y[test_index]

        model = ModelClass(**params).to(device)

        print(f'fold{k} Training ------------')  # 开始训练
        dev_acc = Fit(model, X_train, y_train, validation_data=[
            X_dev, y_dev], criterion=criterion, epochs=epochs, batch_size=batch_size, verbose=verbose, log_fold_name=log_fold_name+f"_cv{k}")

        # 开始测试
        if validation_data:
            print('Evaluating model on test set...')
            X_test, y_test = validation_data
            # scores = Evaluate(model, X_test, y_test)[0]
            scores = Evaluate(model, X_test, y_test)
            print("Result on test set: %s: %.2f%%" % ("acc", scores[0] * 100))
            
            test_score.append(scores[0] * 100)
            
        
        dev_score.append(dev_acc)
        if one_fold:
            return model

    print("CV Dev Result:%.2f%%" % np.mean(dev_score))
    if validation_data:
        print("CV Test Result:%.2f%%" % np.mean(test_score))
    return model
