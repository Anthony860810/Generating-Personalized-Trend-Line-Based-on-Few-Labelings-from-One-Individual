import os
import sys
import argparse
import math
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset

trendfiledir = "./trend/"
directory = "./A4Benchmark/A4Benchmark-TS"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TensorData(Dataset):
    def __init__(self, fileNames):
        self.fileNames = fileNames
    def __len__(self):
        return len(self.fileNames)
    def __getitem__(self, index):
        file_idx = self.fileNames[index]
        x=[]
        y=[]
        value_dir = directory+filename(file_idx)+".csv"
        df = pd.read_csv(value_dir)
        x = np.array(df["value"])
        with open(file) as f:
            data = json.load(f)
            y = np.array(data["trend"][file_idx])
        return x,y

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Info
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
        # out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def filename(index):
    if index==0:
        return str(4)
    elif index==1:
        return str(6)
    elif index==2:
        return str(15)
    elif index==3:
        return str(17)
    elif index==4:
        return str(24)
    elif index==5:
        return str(25)
    elif index==6:
        return str(33)
    elif index==7:
        return str(36)
    elif index==8:
        return str(49)
    elif index==9:
        return str(59)
    elif index==10:
        return str(66)
    elif index==11:
        return str(74)
    elif index==12:
        return str(81)
    elif index==13:
        return str(88)
    elif index==14:
        return str(91)
    elif index==15:
        return "1_v2"
    elif index==16:
        return "21_v2"
    elif index==17:
        return "8_v2"
    elif index==18:
        return "12_v2"
    elif index==19:
        return "20_v2"

def LoadModel():
    model = transformerNet()
    model.to(device, dtype=torch.double)
    # model.load_state_dict(torch.load(model_directory))
    return model


def GetTrend(idx):
    trendfile = trendfiledir+filename(idx)+".json"
    with open (trendfile) as f:
        data = json.load(f)
        value = data["value"]
        l1norm = data["l1norm"]
        hp = data["hp"]
        stl = data["stl"]
    return value, l1norm, hp, stl


def Draw(value, user_trend, simulate_trend, idx, Pattern_Name=""):
    plt.plot(value, color="mediumspringgreen", alpha=0.8)
    # plt.plot(user_trend, linewidth=2, color="blueviolet")
    plt.plot(simulate_trend, linewidth=2, color="peru")
    plt.savefig(img_dir+filename(idx)+"/"+ Pattern_Name+ ".pdf")
    plt.close()

def DrawTrainLoss(loss):
    plt.plot(loss)
    plt.savefig("transformer_TrainLoss.png")
    plt.close()

def SMAPE (simulate_trend, user_trend):
    return torch.mean(2*torch.abs(simulate_trend-user_trend)/(torch.abs(user_trend)+torch.abs(simulate_trend)))

def CaculateMSE(user_trend, simulate_trend):
    loss_function = nn.MSELoss()
    loss = loss_function(torch.tensor(user_trend), torch.tensor(simulate_trend))
    return loss.item()

## Get user trend
def ReadUserTrend():
    with open (file) as f:
        data = json.load(f)
    return data["trend"]

def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--epoch", default = 20, help = "Epoch", type=int)
    _parser.add_argument("--lr", default=0.075, help="Learning rate", type=float)
    _parser.add_argument("--batch", default=10, help="Batch size", type=int)
    _parser.add_argument("--user", default="29", help="User")
    args = _parser.parse_args()
    epochs, lr, batch_size, user = args.epoch, args.lr, args.batch, args.user
    file = './Done_user/user'+user+"/user"+user+".json"
    img_dir = "./s2_img_user"+user+"/"
    return epochs, lr, batch_size, user, file, img_dir

if __name__ == '__main__':
    epochs, lr, batch_size, user, file, img_dir = ParseInput()
    train_idxs = list(range(10))
    test_idxs = list(range(10,20))
    train_tensor = TensorData(train_idxs)
    test_tensor = TensorData(test_idxs)
    train_loader = DataLoader(
        dataset = train_tensor,
        batch_size = batch_size,
        num_workers = 8
    )
    test_loader = DataLoader(
        dataset = test_tensor,
        batch_size = batch_size,
        num_workers = 8
    )
    model = LoadModel()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    torch.cuda.empty_cache()
    model.train()
    idx = 10
    train_loss = []
    for epoch in tqdm(range(epochs)):
        running_loss=0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1 , 1680)
            target = target.view(-1, 1 , 1680)
            y_hat = model(data, target)
            target = target.to(dtype=float)
            loss = loss_func(y_hat, target)
            
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(running_loss)
        print("Epoch_" + str(epoch) + "_loss: " + str(running_loss) )
    ## Test data
    MSE_errors = []
    SMAPE_errors=[]
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1 , 1680)
            target = target.view(-1, 1 , 1680)
            y_hat = model(data, target)
            for i in range(len(target)):
                SMAPE_errors.append(SMAPE(target[i], y_hat[i]).item())
                MSE_errors.append(loss_func(target[i], y_hat[i]).item())
            for i in range (len(data)):
                time_series = data[i].view(-1)
                target_i = target[i].view(-1)
                y_hat_i = y_hat[i].view(-1) 
                Draw(time_series.cpu().detach().numpy(), target_i.cpu().detach().numpy(), y_hat_i.cpu().detach().numpy(), idx, "transformer")
                idx+=1
    print("## Transformer")
    print("User: ", user)
    old_env_MSE = MSE_errors[:5]
    new_env_MSE = MSE_errors[5:]
    old_env_SMAPE = SMAPE_errors[:5]
    new_env_SMAPE = SMAPE_errors[5:]
    print("* Old env MSE: ", old_env_MSE,",")
    print("* New env MSE: ", new_env_MSE,",")
    
    print("* Old env SMAPE: ", old_env_SMAPE,",")
    print("* New env SMAPE: ", new_env_SMAPE,",")
    
    print("* Old env MSE mean: ", str(round(sum(old_env_MSE)/len(old_env_MSE),2)),"±", str(round(np.std(old_env_MSE, ddof=1),2)))
    print("* New env MSE mean: ", str(round(sum(new_env_MSE)/len(new_env_MSE),2)),"±", str(round(np.std(new_env_MSE, ddof=1),2)))
    
    print("* Old env SMAPE mean: ", str(round(sum(old_env_SMAPE)/len(old_env_SMAPE),2)),"±", str(round(np.std(old_env_SMAPE, ddof=1),2)))
    print("* New env SMAPE mean: ", str(round(sum(new_env_SMAPE)/len(new_env_SMAPE),2)),"±", str(round(np.std(new_env_SMAPE, ddof=1),2)))