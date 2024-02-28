import numpy as np 
import matplotlib.pylab as plt
import pandas as pd

Ndata= 1000
	
d1 = np.random.multivariate_normal(mean=[0,2], cov=[[2,-5],[-5,3]], size=Ndata)
d2 = np.random.multivariate_normal(mean=[8,6], cov=[[5,-3],[-3,8]], size=Ndata)
N = Ndata
X1 = np.c_[np.ones([N,1]), d1]
X2 = np.c_[np.ones([N,1]), d2]
X = np.r_[X1,X2]
y1 = np.zeros([N,1])
y2 = np.ones([N,1])
y = np.r_[y1,y2]

plt.scatter(d1[:,0], d1[:,1],c="b")
plt.scatter(d2[:,0], d2[:,1], c="r")
f3 = plt.figure()
ax3 = plt.axes(projection = '3d')
ax3.plot(d1[:,0], d1[:,1], 0, 'go')
ax3.plot(d2[:,0], d2[:,1], 1, 'ro')

def sigmoid(x):
	return 1.0/(1+np.exp(-x))

eta = 0.1
n_iterations =100
wGD = np.zeros([3,1])
wGDbuffer = np.zeros([3,n_iterations+1])

for iteration in range(n_iterations):
	mu = sigmoid(wGD.T.dot(X.T)).T
	gradients = X.T.dot(mu-y)
	wGD = wGD - eta*gradients
	#wGDbuffer[:,iteration+1] = [wGD[0], wGD[1], wGD[2]]


x1sig = np.linspace(-5,10,100)
x2sig = np.linspace(-5,10,100)
print("length of x1sig:",len(x1sig),"length of x2sig ", len(x2sig))
[x1Sig, x2Sig] = np.meshgrid(x1sig, x2sig)
wTx = wGD[1]* x1Sig + wGD[2] * x2sig + wGD[0]
print(len(wTx))
ySig = sigmoid(wTx)

#f5 = plt.figure()
#ax5 = plt.axes(projection = '3d')
ax3.plot_surface(x1Sig, x2Sig, ySig, cmap = 'viridis')
#ax3.contour3D(x1Sig, x2Sig, ySig, 50)
plt.show()