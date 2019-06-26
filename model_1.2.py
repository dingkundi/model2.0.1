from sklearn.externals import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import time

time_one = time.time()
data = pd.read_excel("input4.xlsx",index=None)
data.loc[:,"J"] = 0
for i in range(data.shape[0]-1):
    if data.loc[i,"A"]>data.loc[i+1,"A"]:
        data.loc[i+1,"J"] = 10
data.loc[:,"K"] = data.loc[:,'E']+data.loc[:,"F"]
data1 = data[["H","A","B","C","E","F","N","DATE","TIME","K","J"]]
index = data1[(data1.K==0)].index.tolist()
data2 = data1.drop(index)
data2.loc[:,'A'] = data2.loc[:,'A'].map(lambda x:float(x)*100)
data3 = data2.loc[data2['A']>=80]
data46 = data3.loc[data3['A']<=1000]
index = data46[(data46.H==9)].index.tolist()
for i in index:
    data46.loc[i,"H"] = 0
data46 = data46.fillna(0)
# data46.loc[index,'N'] = 1
# data46.loc[:,"N"].fillna(0,inplace=True)
data46 = data46[['H','A','B','C','E','F','N','DATE','TIME',"J"]]
data46.loc[:,'G'] = 0

bst = joblib.load('./model/best_model_1-4.model')
data = data46.reset_index().drop(columns=["index"])
# print(data)
shape = data.shape
# print(shape)
# count_one = int(shape[0]*0.025)  # 1的总数
X_test = data[['A','B','C']]
dtest = xgb.DMatrix(X_test)
# pred = bst.predict(dtest)
# data.loc[:,'H'] = pred

pred = bst.predict(dtest)
# pred[pred<  0.75] = 0
# pred[pred>= 0.75] = 1

data.loc[:,'H'] = pred
for i in range(shape[0]):
    if data.loc[i,"E"] == 1 and data.loc[i,"H"]>=0.72:
        data.loc[i,"H"] = 1
    elif data.loc[i, "F"] == 1 and data.loc[i, "H"] >= 0.75:
        data.loc[i,"H"] = 1
    else:
        data.loc[i,"H"] = 0


for i in range(shape[0]):
    if data.loc[i,"J"] == 10:
        data.loc[i,"H"] = 0
# index_1 = data.sort_values(by='H', ascending=False).index
# index1_1 = index_1[0:count_one]
# data.loc[:,'H'] = 0
# data.loc[index1_1,'H'] = 1

bst1 = joblib.load('./model/best_model_9-3.model')
X_test1 = data[['A','B','C','H']]
dtest = xgb.DMatrix(X_test1)
pred1 = bst1.predict(dtest)
data.loc[:,'N'] = pred1
data.loc[:,'R'] = 0
data.reset_index()

grp = 1
data.loc[0,"G"] = grp
for i in range(data.shape[0]-1):
    if data.loc[i,"E"] != data.loc[i+1,"E"]:
        grp +=1
    data.loc[i+1,"G"] = grp
# data.loc[data.shape[0]-1,"G"] = data.loc[data.shape[0]-2,"G"]
for i in range(data.shape[0]):
    if data.loc[i, "H"] == 1:
        data.loc[i, "N"] = 0
# data.to_excel("1.xlsx")
max_index = []
count_index = 0
for i in range(1, grp+1):
    index = data[(data.G == i)].index.tolist()
    data1 = data.iloc[index]
    value = np.array(data1.loc[:, "N"])
    max_index1 = np.argmax(value) + count_index
    count_index += len(index)
    max_index.append(max_index1)

data2 = data[(data['H'] == 1)]
value_count = sorted(list(set(data2['G'].values)))
# data.to_excel("111.xlsx")
# print(value_count)
if value_count[-1] == data['G'].max():
    value_count.pop()
# print(value_count)
# print(max_index)
for i in value_count:
    a = int(max_index[int(i)])
    data.loc[a,"R"] = 9
data31 = data[["H","E","F","N","DATE","TIME","R","A"]]


def tran_to_date(x):
    if len(x) == 10:
        a = x[0:2]
        c = x[4:5]
        e = x[8:10]
        if x[2] =='0':
            b = x[3]
        else:
            b = x[2:4]
        if x[4] =='0':
            c = x[5]
        else:
            c = x[4:6]
        if x[6] =='0':
            d = x[7]
        else:
            d = x[6:8]
        time = "20"+a+"/"+b+"/"+c+" "+d+':'+e
        return time

data33 = data31

def str1(x):
    x = str(int(x))
    if len(x) == 5:
        x = "0"+x
    return x


data33.loc[:,'A'] = data33.loc[:,'A'].map(lambda x:float(x)/100)
data33.loc[:,'DATE'] = data33.loc[:,'DATE'].map(lambda x:str(x)[0:7])
data33.loc[:,'TIME'] = data33.loc[:,'TIME'].map(lambda x:str1(x))
data33.loc[:,'time'] = data33.loc[:,'DATE']+data33.loc[:,'TIME']


data4 = data33.sort_values(by="time" , ascending=True)
data4['time'] = data4['time'].map(lambda x:str(x)[1:-2])
data4['time'] = data4['time'].map(lambda x:tran_to_date(x))
data5 = data4[['time','H','R','E','F','A']]
data5 = data5.fillna('0')
data5.loc[:,'R'] = data5.loc[:,'R'].map(lambda x:float(x))
data5.loc[:,'H'] = data5.loc[:,'H'].map(lambda x:float(x))
data5.loc[:,'H'] = data5.loc[:,'R']+data5.loc[:,'H']
data5.loc[:,'R'] = data5.loc[:,'H']
data5.to_csv('result.csv',index=None)
time_two = time.time()
all_time = str(time_two-time_one)
data_show = data5











































































