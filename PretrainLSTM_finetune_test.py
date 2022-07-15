import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
trendfiledir = "./trend/"
directory = "./A4Benchmark/A4Benchmark-TS"
model_directory = "pretrainedlstm_model"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class TensorData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class lstmNet(nn.Module):
    def __init__(self):  # override __init__
        super(lstmNet, self).__init__() # 使用父class的__init__()初始化網路
        self.layer1 = nn.LSTM(1680,2048)
        self.activate = nn.ReLU()
        self.layer4 = nn.LSTM(2048,1680)
        self.fc = nn.Linear(1680,1680)
    def forward(self, x):
        out, (hn, cn) = self.layer1(x)
        out = self.activate(out)
        out, (hn, cn) = self.layer4(out)
        out = self.activate(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def LoadModel():
    model = lstmNet()
    model.to(device, dtype=torch.double)
    model.load_state_dict(torch.load(model_directory))
    count=0
    # print(model.parameters())
    for layer in model.children():
        if count < 3:
            for param in layer.parameters():
                param.requires_grad = False
            count+=1
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
    plt.savefig("TrainLoss_pretrainlstm5.pdf")
    plt.close()

def DrawTrainTestLoss(train_loss, test_loss):
    plt.plot(train_loss, color="orange")
    plt.plot(test_loss, color="mediumorchid")
    plt.savefig("TrainTestCompareLoss_pretrainlstm5.pdf")
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

def TrainTestLoad(train_idxs, test_idxs):
    X_train, X_test, Y_train, Y_test =[],[],[],[]
    ### train data
    for file_idx in train_idxs:
        value_dir = directory+filename(file_idx)+".csv"
        df = pd.read_csv(value_dir)
        x = np.array(df["value"])
        X_train.append(x)
        with open(file) as f:
            data = json.load(f)
            y = np.array(data["trend"][file_idx])
            Y_train.append(y)
    ### test data
    for file_idx in test_idxs:
        value_dir = directory+filename(file_idx)+".csv"
        df = pd.read_csv(value_dir)
        x = np.array(df["value"])
        X_test.append(x)
        with open(file) as f:
            data = json.load(f)
            y = np.array(data["trend"][file_idx])
            Y_test.append(y)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test

def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--epoch", default = 5, help = "Epoch", type=int)
    _parser.add_argument("--lr", default=0.05, help="Learning rate", type=float)
    _parser.add_argument("--batch", default=10, help="Batch size", type=int)
    _parser.add_argument("--user", default="29", help="User")
    args = _parser.parse_args()
    epochs, lr, batch_size, user = args.epoch, args.lr, args.batch, args.user
    file = './user'+user+"/user"+user+".json"
    img_dir = "./s2_img_user"+user+"/"
    return epochs, lr, batch_size, user, file, img_dir



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, filename="pretrainlstmLog.log", filemode='w', format=FORMAT)
    epochs, lr, batch_size, user, file, img_dir = ParseInput()
    
    train_idxs = list(range(10))
    test_idxs = list(range(10,20))
    X_train, X_test, Y_train, Y_test = TrainTestLoad(train_idxs, test_idxs)
    train_tensor = TensorData(X_train, Y_train)
    test_tensor = TensorData(X_test, Y_test)
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
    optimizer = torch.optim.Adamax(model.fc.parameters(), lr=lr)
    torch.cuda.empty_cache()
    model.train()
    idx = 10
    train_loss = []
    test_loss=[]
    for epoch in tqdm(range(epochs)):
        model.train()
        train_running_loss=0.0
        test_running_loss=0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1 , 1680)
            y_hat = model(data)
            target = target.to(dtype=float)
            loss = loss_func(y_hat, target)
            train_running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch_" + str(epoch) + "_loss: " + str(train_running_loss) )
        train_loss.append(train_running_loss)
    # DrawTrainLoss(train_loss)
    ## Test data
    MSE_errors = []
    SMAPE_errors=[]
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 1 , 1680)
                y_hat = model(data)
                test_loss.append(loss_func(y_hat, target).item())
                for i in range(len(target)):
                    SMAPE_errors.append(SMAPE(target[i], y_hat[i]).item())
                    MSE_errors.append(loss_func(target[i], y_hat[i]).item())
                for i in range (len(data)):
                    time_series = data[i].view(-1)
                    Draw(time_series.cpu().detach().numpy(), target[i].cpu().detach().numpy(), y_hat[i].cpu().detach().numpy(), idx, "pretrainlstm")
                    idx+=1
    # DrawTrainTestLoss(train_loss, test_loss)
    print("## pretrainLSTM")
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