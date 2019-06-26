import pandas as pd
import numpy as np


def dire(x):
    if x == 0:
        return -1
    if x == 1:
        return 1
data = pd.read_excel("data.xlsx")
data.loc[:,'direct'] = data.loc[:,'E'].map(lambda x:dire(x))


index1 = data[(data.H == 1)].index.tolist()
index9 = data[(data.H == 9)].index.tolist()
D1 = 0
for i in range(len(index1)):
    start,stop = index1[i],index9[i]
    # print(start,stop)
    data.loc[stop, 'X9-X1'] = data.loc[stop, 'F1']-data.loc[start, 'F1']
    data.loc[stop, 'oldC'] = (data.loc[stop, 'F1'] - data.loc[start, 'F1'])*data.loc[stop, 'direct']
    D1 = data.loc[stop, 'oldC'] + D1
    data.loc[stop, 'D1'] = D1
    for j in range(stop-start+1):
        data.loc[start+j, 'newX9-X1'] = data.loc[start+j, 'F1'] - data.loc[start, 'F1']
        data.loc[start+j, 'newC'] = (data.loc[start+j, 'F1'] - data.loc[start, 'F1']) * data.loc[start+j, 'direct']
data.loc[:,"D1"].fillna(-100,inplace=True)
D = round(max(data.loc[:,"D1"]),4)
E = abs(round(min(min(data.loc[:,"newC"]),0),4))
F = data.shape[0]
FF = round(D*D1/(D-D1+E)*F,4)
print(FF)
data.to_excel("111.xlsx",index=None)






































































