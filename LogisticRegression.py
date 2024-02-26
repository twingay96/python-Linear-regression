# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pylab as plt
import pandas as pd

dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LogisticRegression.txt", sep="\s+")
xxRaw = np.array(dfLoad.values[:,0])
yyRaw = np.array(dfLoad.values[:,1])
plt.plot(xxRaw, yyRaw, "k.")


def sigmoid(x):
	return 1.0/(1+np.exp(-x))
#xxTest = np.linspace(-10, 10, num=101)
#plt.plot(xxTest, sigmoid(xxTest), "k-")

N = len(xxRaw)
x_bias = np.c_[np.ones([N,1]), xxRaw].T #Padding ones for X0
y = yyRaw.reshape(N,1)
X = x_bias.T

eta = 0.1
n_iterations = 1000
wGD = np.zeros([2,1]) # initialized to 0,0 2x1 행렬
wGDbuffer = np.zeros([2,n_iterations+1])

print("wGD 초기값:", wGD )

for iteration in range(n_iterations):
    mu = sigmoid(wGD.T.dot(x_bias)).T
    gradients = X.T.dot(mu-y)
    wGD = wGD - eta*gradients
    # 매번 gradient된 결과를 하나씩 iteration갯수에 따라서 넣는 것
    #wGDbuffer[:,iteration+1] = [wGD[0], wGD[1]]
    
print(sigmoid(3))
xxTest = np.linspace(0,10,num = N).reshape(N,1)
xxTest_bias = np.c_[np.ones([N,1]),xxTest]
aaa = sigmoid(wGD.T.dot(xxTest_bias.T))
plt.plot(xxTest, sigmoid(wGD.T.dot(xxTest_bias.T)).T, "r-.")
#plt.show()
