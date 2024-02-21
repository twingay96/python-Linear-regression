import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dfLoad = pd.read_csv("https://sites.google.com/site/vlsicir/testData_LinearRegression.txt",sep="\s+")
xxRaw = dfLoad["xx"]
yyRaw = dfLoad["yy"]
yyRaw = np.array(yyRaw)
plt.plot(xxRaw, yyRaw, "r.")

