import numpy as np
import matplotlib.pylab as plt
import pandas as pd

Ndata = 1000
A = np.random.randn(Ndata)
f1 = plt.figure()
ax1 = plt.axes()
ax1.hist(A,bins=10)
plt.show()

f2= plt.figure()
ax2 = plt.axes()
d1 = np.random.multivariate_normal(mean=[0,2], cov=[[2,-5],[-5,3]], size=Ndata)
d2 = np.random.multivariate_normal(mean=[8,6], cov=[[5,-3],[-3,8]], size=Ndata)

f3 = plt.figure()
ax3 = plt.axes(projection = '3d')
ax3.plot(d1[:,0],d1[:,1],0, 'go')
ax3.plot(d2[:,0],d2[:,1],0, 'ro')
plt.show()




