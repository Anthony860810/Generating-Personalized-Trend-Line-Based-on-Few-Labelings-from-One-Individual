import argparse
import json
import matplotlib.pyplot as plt
import os

load_dir = "./trend/"
save_l1 = "l1"
save_hp = "hp"
save_stl = "stl"

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


def Draw(value, simulate_trend, idx, Pattern_Name="", img_dir=""):
    plt.plot(value, color="mediumspringgreen", alpha=0.8)
    plt.plot(simulate_trend, linewidth=2, color="peru")
    plt.savefig(img_dir+filename(idx)+"/"+Pattern_Name+ ".pdf")
    plt.close()

def ParseInput():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--user", default="29", help="User")
    args = _parser.parse_args()
    user = args.user
    img_dir = "./s2_img_user"+user+"/"
    return img_dir

if __name__ == '__main__':
    img_dir = ParseInput()
    for i in range(11,20):
        sourcedata = load_dir+filename(i)+".json"
        with open(sourcedata) as f:
            json_file = json.load(f)
            Draw(json_file["value"], json_file["l1norm"], i, save_l1, img_dir)
            Draw(json_file["value"], json_file["hp"], i, save_hp, img_dir)
            Draw(json_file["value"], json_file["stl"], i, save_stl, img_dir)