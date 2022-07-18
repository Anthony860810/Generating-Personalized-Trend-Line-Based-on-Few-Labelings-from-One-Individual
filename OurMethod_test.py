import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


trendfiledir = "./trend/"
directory = "./A4Benchmark/A4Benchmark-TS"
model_directory = "OurMethod_model"
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

## Transform into Tensor
class TensorData(Dataset):
    def __init__(self, fileNames):
        self.fileNames = fileNames
    def __len__(self):
        return len(self.fileNames)
    def __getitem__(self, index):
        file_idx = self.fileNames[index]
        x=[]
        value_dir = directory+filename(file_idx)+".csv"
        df = pd.read_csv(value_dir)
        value = df["value"]
        with open(file) as f:
            data = json.load(f)
            trend = np.array(data["trend"][file_idx])
            x.append(np.vstack((value,trend)))
        x = torch.tensor(np.array(x))
        x = x.view(2,1680)
        return x

def MergeData(trend_data):
    merge_data = {}
    orginal_data = []
    for index in range(len(trend_data["trend"])):
        name = directory+filename(index)+".csv"
        orginal_data.append(name)
    merge_data["trend"]=trend_data["trend"]
    merge_data["value_dir"]=orginal_data
    return merge_data


## Get user trend
def ReadUserTrend():
    with open (file) as f:
        data = json.load(f)
    return data["trend"]

## Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(107008, 3),
            nn.BatchNorm1d(3)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        out = nn.Softmax()(out)
        return out

def LoadModel():
    model = ConvNet()
    model.to(device, dtype=torch.double)
    model.load_state_dict(torch.load(model_directory))
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


def SMAPE (simulate_trend, user_trend):
    return torch.mean(2*torch.abs(simulate_trend-user_trend)/(torch.abs(user_trend)+torch.abs(simulate_trend)))

def AverageUserPattern(user_patterns):
    return np.sum(user_pattern, axis=0)/10

def WeightsUserPattern(simulate_trends, user_trends, user_pattern):
    inverse_SMAPE_ratio=[]
    for idx in range(len(simulate_trends)):
        ## SMAPE smaller is closer, so inverse's one bigger means more important
        inverse_SMAPE_ratio.append(1/SMAPE(torch.tensor(simulate_trends[idx]), torch.tensor(user_trends[idx])).item())
    # print("Inverse SMAPE ratios: ",inverse_SMAPE_ratio)
    sum_inverse_SMAPE = sum(inverse_SMAPE_ratio)
    percentage_predict_ratio = np.array(inverse_SMAPE_ratio)/sum_inverse_SMAPE
    # print("Precentage: ", percentage_predict_ratio)
    l1=0
    hp=0
    stl=0
    for i in range(len(percentage_predict_ratio)):
        l1+=(percentage_predict_ratio[i]*user_pattern[i][0])
        hp+=(percentage_predict_ratio[i]*user_pattern[i][1])
        stl+=(percentage_predict_ratio[i]*user_pattern[i][2])
    return [l1, hp, stl]

def Evaluation_RealWorldTest(user_pattern, user_trends, Pattern_name):
    ## 模擬實際場景定義好使用者行為後做的測試
    test_idx = list(range(10,20))
    index=10
    MSE_errors=[]
    SMAPE_errors =[]
    for idx in test_idx:
        value, l1_norm, hp, stl = GetTrend(idx)
        simulate_trend = np.array(l1_norm)*user_pattern[0] + np.array(hp)*user_pattern[1] + np.array(stl)*user_pattern[2]
        Draw(value, user_trends[index], simulate_trend, idx, Pattern_name)
        MSE_errors.append(CaculateMSE(user_trends[index], simulate_trend))
        SMAPE_errors.append(SMAPE(torch.tensor(simulate_trend), torch.tensor(user_trends[index])).item())
        index+=1
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


def Evaluation_Backtesting(user_pattern, user_trends, Pattern_name):
    ## 用定義好的使用者行為再回推定義好的行為是否表現不錯
    test_idx = list(range(10))
    index=0
    MSE_errors=[]
    SMAPE_errors =[]
    for idx in test_idx:
        value, l1_norm, hp, stl = GetTrend(idx)
        simulate_trend = np.array(l1_norm)*user_pattern[0] + np.array(hp)*user_pattern[1] + np.array(stl)*user_pattern[2]
        # Draw(value, user_trends[index], simulate_trend, idx, Pattern_name)
        MSE_errors.append(CaculateMSE(user_trends[index], simulate_trend))
        SMAPE_errors.append(SMAPE(torch.tensor(simulate_trend), torch.tensor(user_trends[index])).item())
        index+=1
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



def  Two_Ways_RealWorldTest_Evaluation(Average_pattern, Weights_pattern, user_trends):
    ## Define User pattern (Average)
    print("\n=============Average pattern RealWorldTest Evaluation=============")
    print("Average pattern: ", Average_pattern)
    Evaluation_RealWorldTest(Average_pattern, user_trends, "Average")
    ## Define User pattern (Weight Average) use prior ten data file
    print("\n=============Weights pattern RealWorldTest Evaluation=============")
    print("weight user pattern: ", Weights_pattern)
    Evaluation_RealWorldTest(Weights_pattern, user_trends, "Weights")

def  Two_Ways_Backtesting_Evaluation(Average_pattern, Weights_pattern, user_trends):
    ## Define User pattern (Average)
    print("\n=============Average pattern Backtesting Evaluation=============")
    print("Average pattern: ", Average_pattern)
    Evaluation_Backtesting(Average_pattern, user_trends, "_Backtesting_A")
    ## Define User pattern (Weight Average) use prior ten data file
    print("\n=============Weights pattern Backtesting Evaluation=============")
    print("weight user pattern: ", Weights_pattern)
    Evaluation_Backtesting(Weights_pattern, user_trends, "_Backtesting_W")

def Situation1_Evaluation(simulate_trends, user_trends, values):
    print("\n=============Situation1 Evaluation=============")
    MSE_errors=[]
    SMAPE_errors=[]
    for i in range(len(simulate_trends)):
        MSE_errors.append(CaculateMSE(simulate_trends[i], user_trends[i]))
        SMAPE_errors.append(SMAPE(torch.tensor(simulate_trends[i]), torch.tensor(user_trends[i])).item())
    print("MSE errors: ",MSE_errors)
    print("SMAPE errors: ",SMAPE_errors)
    print("MSE mean: ", str(round(sum(MSE_errors)/len(MSE_errors),2)),"±", str(round(np.std(MSE_errors, ddof=1),2)))
    print("SMAPE mean: ", str(round(sum(SMAPE_errors)/len(SMAPE_errors),4)),"±", str(round(np.std(SMAPE_errors, ddof=1),4)))


def CaculateMSE(user_trend, simulate_trend):
    loss_function = nn.MSELoss()
    loss = loss_function(torch.tensor(user_trend), torch.tensor(simulate_trend))
    return loss.item()

def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--batch", default=10, help="Batch size", type=int)
    _parser.add_argument("--user", default="29", help="User")
    args = _parser.parse_args()
    batch_size, user = args.batch, args.user
    file = './Done_user/user'+user+"/user"+user+".json"
    img_dir = "./s2_img_user"+user+"/"
    return batch_size, user, file, img_dir

if __name__ == '__main__':
    batch_size, user, file, img_dir = ParseInput()
    # 只抓前十個做預測
    idxs = list(range(10))
    test_tensor = TensorData(idxs)
    test_loader = DataLoader(
        dataset = test_tensor,
        batch_size = batch_size,
        num_workers = 8
    )
    model = LoadModel()
    user_pattern = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            y_hat = model(data)
            for i in range(len(y_hat)):
                # print("Predict three parameters: " + str(y_hat[i].cpu().detach().numpy()) + "\n")
                user_pattern.append(y_hat[i].cpu().detach().numpy())
    user_trends = ReadUserTrend()
    ## Situation1:Predict and Generate simulate trend
    simulate_trends = []
    values=[]
    for idx in range(len(user_pattern)):
        value, l1_norm, hp, stl = GetTrend(idx)
        values.append(value)
        simulate_trend = np.array(l1_norm)*user_pattern[idx][0] + np.array(hp)*user_pattern[idx][1] + np.array(stl)*user_pattern[idx][2]
        simulate_trends.append(simulate_trend)
        # Draw(value, user_trends[idx], simulate_trend, idx)
    #Situation1_Evaluation(simulate_trends, user_trends, values)
    Average_pattern = AverageUserPattern(user_pattern)
    Weights_pattern = WeightsUserPattern(simulate_trends, user_trends, user_pattern)
    Two_Ways_RealWorldTest_Evaluation(Average_pattern, Weights_pattern, user_trends)
    # Two_Ways_Backtesting_Evaluation(Average_pattern, Weights_pattern, user_trends)