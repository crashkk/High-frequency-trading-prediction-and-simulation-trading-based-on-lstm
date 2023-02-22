from pandas import *
import os
import numpy as np

def read_data(dir_name):#读取excell股票数据,返回dataframe
    excellFilePath='{}'.format(dir_name)
    for filename in os.listdir(excellFilePath):
        dele=[]
        filepath='{}/{}'.format(excellFilePath,filename)
        dataf=read_excel(filepath,sheet_name=0,names=list('1234567'),header=None,index_col=0)
        for i in range(1,dataf.shape[0]-1):
            if dataf.loc[i][3]==0:
                dele.append(i)
        dele.append(0)
        dataf=dataf.drop(dele)#axis 0表示行，1表示列
        dataf.to_excel('data/shice_new/'+filename)
        print('{} process down'.format(filename))

read_data('data/shice')
"""
#read_data('data/old_data/002594.SZ.xlsx')
dele=[]
dataf=read_excel('data/old_data/002594.SZ.xlsx',sheet_name=0,names=list('1234567'),header=None,index_col=0)
for i in range(1,dataf.shape[0]-1):
    if dataf.loc[i][3]==0:
        dele.append(i)
dele.append(0)
dataf=dataf.drop(dele)#axis 0表示行，1表示列
dataf.to_excel('data/new_data/'+'002594.SZ.xlsx')
print('{} process down'.format('002594'))
"""