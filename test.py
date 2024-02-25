# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math as m
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

plt.close("all")
dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LinearRegression.txt"
                     ,sep="\s+")
xxRaw = dfLoad["xx"]
yyRaw = dfLoad["yy"]
yyRawNP = np.array(yyRaw)
plt.plot(xxRaw,yyRaw, "r.")

#Normal Equation
Ndata = len(xxRaw)
X = np.c_[np.ones([Ndata, 1]),xxRaw]
wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yyRawNP.reshape(Ndata,1))
xPredict = np.linspace(0,2,num=101)
xPredictPadding = np.c_[np.ones([101,1]), xPredict]
yPredict = wOLS.T.dot(xPredictPadding.T)

plt.plot(xPredict.reshape(1,101),yPredict, "b.-")
