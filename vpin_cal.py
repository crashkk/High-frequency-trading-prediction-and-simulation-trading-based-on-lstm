from pandas import *
import os
import numpy as np
import scipy.io
import csv
from feature_func import totalfunc
from collections import Counter
import matplotlib.pyplot as plt 
from math import *
fun=totalfunc()
file_name='data/old_data/601788.SH_min.xlsx'
data=read_excel(file_name,sheet_name=0)
data=np.array(data)
header=data[0,:]
data=np.delete(data,0,0)
time_list=data[:,0]
data=np.delete(data,0,1)
data=data[2650:2771,:]
cal=0
P=np.zeros((5))#存储PEi-PBi
V=np.zeros((5))#存储成交量
VB=np.zeros((5))
VS=np.zeros((5))
vpin=[]
Vpin=np.zeros((data.shape[0],2))
for i in range(data.shape[0]):
    if cal<=4:
        P[cal]=data[i,0]-data[i,1]
        V[cal]=data[i,-1]
        cal+=1
    if cal==5:
        V_total=np.sum(data[i-4:i+1,-1])*100#这五分钟的总交易量
        P=fun.standardization(P)#标准化
        VB=np.multiply(V,P)
        print(V)
        VS=np.multiply(V,1-P)
        VS=V_total*VS-VB
        answer=np.sum(abs(VS-VB)/V_total)/5
        vpin.append(answer)
        Vpin[i,0]=i
        Vpin[i,1]=answer
        cal=0
print(Vpin.shape[0])
Vpin=Vpin[Vpin[:,0]!=0,:]/1000
print(Vpin)
fig=plt.figure()
axes1 = fig.add_subplot(2,1,1)
plt.xlabel('time')
plt.ylabel('avg price')
plt.plot(np.arange(data.shape[0]-1),data[1:,-2]/data[1:,-1]/100,color='red')

axes2 = fig.add_subplot(2,1,2)
plt.xlabel('time')
plt.ylabel('vpin')
plt.plot(Vpin[:,0],Vpin[:,1],color='blue')
plt.show()
