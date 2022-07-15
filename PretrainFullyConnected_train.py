import os
import argparse
import json
import sklearn
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
from sklearn.model_selection import train_test_split


directory = "./mixer_multiple_full/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FORMAT = '%(asctime)s %(levelname)s: %(message)s'


class TensorData(Dataset):
    def __init__(self, fileNames):
        self.fileNames = fileNames
    def __len__(self):
        return len(self.fileNames)
    def __getitem__(self, index):
        file = directory+self.fileNames[index]
        x=[]
        y=[]
        with open(file) as f:
            data = json.load(f)
            x = np.array(data["value"])
            y = np.array(data["trend"])
        return x,y

class fcNet(nn.Module):
    def __init__(self):  # override __init__
        super(fcNet, self).__init__() # 使用父class的__init__()初始化網路
        self.layer1 = nn.Linear(1680,1680)
        self.activate = nn.ReLU()
        self.layer2 = nn.Linear(1680,1680)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


def LoadModel():
    model = fcNet()
    model.to(device, dtype=torch.double)
    return model

def DrawTrainLoss(loss):
    plt.plot(loss)
    plt.savefig("TrainLoss_fcpretrain.png")
    plt.close()

def TrainValidationSplit(x):
    X_train, X_validation = train_test_split(x, test_size=0.3, random_state=50)
    return  X_train, X_validation

def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--epoch", default = 5000, help = "Epoch", type=int)
    _parser.add_argument("--lr", default=0.05, help="Learning rate", type=float)
    _parser.add_argument("--batch", default=1000, help="Batch size", type=int)
    args = _parser.parse_args()
    epochs, lr, batch_size = args.epoch, args.lr, args.batch
    return epochs, lr, batch_size

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, filename="pretrainfcLog.log", filemode='w', format=FORMAT)
    epochs, lr, batch_size = ParseInput()
    x = os.listdir(directory)
    for i in x:
        if i[-4:]!="json":
            x.remove(i)
    train, validation = TrainValidationSplit(x)
    train_tensor = TensorData(train)
    train_loader = DataLoader(
        dataset = train_tensor,
        batch_size = batch_size,
        num_workers = 8
    )
    validation_tensor = TensorData(validation)
    validation_loader = DataLoader(
        dataset = validation_tensor,
        batch_size = batch_size,
        num_workers = 8
    )
    model = LoadModel()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    torch.cuda.empty_cache()
    best_model = model
    best_model_epoch=0
    model.train()
    train_loss = []
    min_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        train_running_loss=0.0
        validation_running_loss=0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            y_hat = model(data)
            target = target.to(dtype=float)
            loss = loss_func(y_hat, target)
            train_running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logging.info("Train -> Epoch_" + str(epoch) + "_loss: " + str(train_running_loss) )
        train_loss.append(train_running_loss)
        ### Validation
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_loader):
                data, target = data.to(device), target.to(device
                y_hat = model(data)
                target = target.to(dtype=float)
                loss = loss_func(y_hat, target)
                validation_running_loss+=loss.item()
            logging.info("Validation -> Epoch_" + str(epoch) + "_loss: " + str(validation_running_loss) )
            if validation_running_loss < min_loss:
                min_loss = validation_running_loss
                best_model = model
                best_model_epoch = epoch
    DrawTrainLoss(train_loss)
    torch.save(best_model.state_dict(),"pretrainedfc_model")
    logging.info("Best model epoch: "+str(best_model_epoch))
    