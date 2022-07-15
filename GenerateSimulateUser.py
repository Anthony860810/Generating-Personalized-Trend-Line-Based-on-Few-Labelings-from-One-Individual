import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import cvxpy as cp
import scipy
import cvxopt
import os
import matplotlib.pyplot as plt
import random
import json

DataPath = "./A4Benchmark/"
OutputPath = "./mixer_multiple_full/"

count=0
weight=200000
stl_period = 30
f = open('output_multiple.txt', 'w')

for fileName in os.listdir(DataPath):
    if(fileName[0:14]=="A4Benchmark-TS"):
        print(fileName)
        file = os.path.join(DataPath,fileName)
        df = pd.read_csv(file)
        y = df['value'].to_numpy()
        ## hpfilter
        cycle, hpfilter_trend = sm.tsa.filters.hpfilter(y,weight)

        ## l1norm
        ones_row = np.ones((1, len(y)))
        D = scipy.sparse.spdiags(np.vstack((ones_row, -2*ones_row, ones_row)), range(3), len(y)-2, len(y))
        
        y_hat = cp.Variable(shape=len(y))
        objective_func = cp.Minimize(0.5*cp.sum_squares(y-y_hat)+weight*cp.norm(D@y_hat, 1))
        problem = cp.Problem(objective_func)
        problem.solve(verbose=False)
        l1norm_trend = np.array(y_hat.value)
        
        ## stl
        stl_res = STL(y, period=stl_period, robust=False).fit()

        for i in np.arange(0, 1.01, 0.01):
            l1_weight = abs(round(i,2))
            for j in np.arange(0, round(1.01-l1_weight,2), 0.01):
                hp_weight = abs(round(j,2))
                stl_weight = abs(round(1-l1_weight-hp_weight,2))

                f.write(fileName+"\n")
                f.write(("l1_weight: " + str(l1_weight) + "\thp_weight: "+ str(hp_weight) + "\tstl_weight: "+ str(stl_weight)+"\n"))
                f.write("-----------------------------------------------\n")
                weights = [l1_weight, hp_weight, stl_weight]
                
                trend = hp_weight*hpfilter_trend + l1_weight*l1norm_trend + stl_weight*stl_res.trend
                ## draw plot
                # plt.plot(df["value"], label="value")
                # plt.plot(trend, label="trend")
                # plt.savefig(OutputPath2+str(count)+".png")
                # plt.clf()
                # plt.close()
            
                output = {
                            "value":y.tolist(),
                            "trend":trend.tolist(),
                            "weights": weights,
                            "penalty":weight
                        }
                with open (OutputPath+str(count)+".json","w") as file:
                    json.dump(output, file)   
                count+=1
f.close()