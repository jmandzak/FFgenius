import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import numpy as np

QBs = pd.read_csv("stats/filteredQBs.csv")
RBs = pd.read_csv("stats/filteredRBs.csv")
WRs = pd.read_csv("stats/filteredWRs.csv")
TEs = pd.read_csv("stats/filteredTEs.csv")
DEFs = pd.read_csv("stats/filteredDEFs.csv")
Ks = pd.read_csv("stats/filteredKs.csv")

all_dfs = [QBs, RBs, WRs, TEs, DEFs, Ks]
titles = ["QBs", "RBs", "WRs", "TEs", "DEFs", "Ks"]

for df, title in zip(all_dfs, titles):
    corr = df.corr()
    corr = np.abs(corr)

    dataplot = sb.heatmap(corr[['Points']].sort_values(by=['Points'], ascending=False), cmap="coolwarm", annot=True)
    plt.title(title)
    plt.show()
    plt.close()