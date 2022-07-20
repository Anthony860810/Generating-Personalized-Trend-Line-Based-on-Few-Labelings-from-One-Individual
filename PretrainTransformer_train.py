import os
import argparse
import json
import sklearn
import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class transformerNet(nn.Module):
    def __init__(self):  # override __init__
        super(transformerNet, self).__init__() # 使用父class的__init__()初始化網路
        self.positional_encoder = PositionalEncoding(
            dim_model=1680, dropout_p=0, max_len=5000
        )
        self.layer1 = nn.Transformer(d_model=1680, nhead=5, num_encoder_layers=1,  batch_first = True)
        self.fc = nn.Linear(1680,1680)
    def forward(self, src, tgt):
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        out = self.layer1(src, tgt)

        out = self.fc(out)
        return out

def LoadModel():
    model = transformerNet()
    model.to(device, dtype=torch.double)
    return model

def DrawTrainLoss(loss):
    plt.plot(loss)
    plt.savefig("TrainLoss_transformerpretrain.png")
    plt.close()

def TrainValidationSplit(x):
    X_train, X_validation = train_test_split(x, test_size=0.3, random_state=50)
    return  X_train, X_validation

def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--epoch", default = 50, help = "Epoch", type=int)
    _parser.add_argument("--lr", default=0.05, help="Learning rate", type=float)
    _parser.add_argument("--batch", default=1000, help="Batch size", type=int)
    args = _parser.parse_args()
    epochs, lr, batch_size = args.epoch, args.lr, args.batch
    return epochs, lr, batch_size

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, filename="pretraintransformerLog.log", filemode='w', format=FORMAT)
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
            data = data.view(-1, 1 , 1680)
            target = target.view(-1, 1 , 1680)
            y_hat = model(data, target)
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
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 1 , 1680)
                target = target.view(-1, 1 , 1680)
                y_hat = model(data, target)
                target = target.to(dtype=float)
                loss = loss_func(y_hat, target)
                validation_running_loss+=loss.item()
            logging.info("Validation -> Epoch_" + str(epoch) + "_loss: " + str(validation_running_loss) )
            if validation_running_loss < min_loss:
                min_loss = validation_running_loss
                best_model = model
                best_model_epoch = epoch
    # DrawTrainLoss(train_loss)
    torch.save(best_model.state_dict(),"pretrainedtransformer_model")
    logging.info("Best model epoch: "+str(best_model_epoch))