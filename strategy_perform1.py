from pandas import *#模拟整个高频交易过程，并给出
from dataprocess import *
import os
import numpy as np
import scipy.io
import csv
import tensorflow as tf
from loss_history import LossHistory
from feature_func import totalfunc
from collections import Counter
from keras.layers.core import Activation, Dense, SpatialDropout1D
from keras.layers import Embedding
from keras.layers import LSTM,Flatten,Dense
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataprocess import csv_file_read
from strategy_perform2 import drawback
import random
import matplotlib.pyplot as plt 
from keras.models import load_model
fun=totalfunc()
ind=1#补充的收盘价
suppliment=[96.63,38.37]#补充数据(300014.SZ,601318.SH)

time_width=3
feature_num=17#样本特征数
p1=0.725#置信度参数(best p=0.61 for000049;best p=0.675 time_step=3 for 300014;best p=0.72 time_step=3 for 601318)
p2=0.725
deal=5#自定义每次的成交量
model = load_model('model_save/601318model.h5')
def predict(pred_data):#模型给出预测。接收参数必须为ndarry的形式,1*time_width*features_num的一个三维张量
    pred=model.predict(pred_data)
    pred=np.squeeze(pred)
    pred=pred.tolist()
    
    prob=max(pred[0],pred[1])
    hint=pred.index(prob)+1#1:上涨,2:下跌
    return hint,prob#发生概率最大事件提示，和其概率

#模拟高频交易。接收的数组应该为包装好的二维数组(time_width*features_num)    
def simulate(data):#将csv中的数据进行处理(这里接收到的数据不应该包含)，每个tick添加上各自的因子，重新构造样本预测序列
    data=data.astype("f8")
    start_pos=data[0,0]
    sig=-1
    total_samp=np.zeros((10,10))
    samp=np.zeros((10,10))
    l=set(list(data[:,0]))
    diff_day=len(l)#统计该数据集涉及到的天数,便于后续分开预测
    pred_data=np.zeros((time_width,feature_num))#初始化预测矩阵
    time_cal=0#时间窗滑动
    earning_rate_record=[]#记录收益率
    earnig=[]#记录每次收益
    db=[]
    win=0#记录胜率
    lose=0
    total_deal=0
    pos_curr_sum=0
    neg_curr_sum=0
    cur=0
    l=list(l)
    l.sort()
    for i in range(diff_day):
        current_slice=data[data[:,0]==l[i],:]
        db.append(drawback(current_slice[:,1]))
        if i==diff_day-1:
            shoupan=suppliment[ind]
        else:
            shoupan=data[data[:,0]==l[i+1],3]
            shoupan=shoupan[0]
        j=0
        while(j<current_slice.shape[0]):
            if j>=23:#23t之后才有ma23特征，因此从23t开始
                if j==23:
                    paras=[50,50]
                if time_cal<=time_width-1:
                    jia_ma_6=fun.jiage_MA(current_slice[j-23:j+1,1],6)#增加新特征
                    jia_ma_12=fun.jiage_MA(current_slice[j-23:j+1,1],12)
                    jia_ma_24=fun.jiage_MA(current_slice[j-23:j+1,1],24)
                    cjl_ma_6=fun.chengjiaoliang_MA(current_slice[j-23:j+1,4],6)
                    cjl_ma_12=fun.chengjiaoliang_MA(current_slice[j-23:j+1,4],12)
                    cjl_ma_24=fun.chengjiaoliang_MA(current_slice[j-23:j+1,4],24)
                    pclimb=fun.p_climb_rate(current_slice[j-23:j+1,1])
                    cclimb=fun.c_climb_rate(current_slice[j-23:j+1,4])
                    K,D,J=fun.KDJ(current_slice[j-23:j+1,1],shoupan,paras)
                    paras=[K,D]
                    pred_data[time_cal,:]=np.array([current_slice[j,1],current_slice[j,4],current_slice[j,5],current_slice[j,6],#为该时刻写入特征
                        jia_ma_6,jia_ma_12,jia_ma_24,cjl_ma_6,cjl_ma_12,cjl_ma_24,pclimb,cclimb,K,D,J,current_slice[j,7],current_slice[j,8]])
                    time_cal+=1
                else:
                    time_cal=0
                    pred_data=fun.normalization(pred_data)
                    predictor,prob=predict(np.expand_dims(pred_data,axis=0))
                    
                    """
                    #对照实验，乱投
                    j_kai=j
                    j+=time_width+1
                    if j>current_slice.shape[0]-1:
                        break
                    current_earn_rate=fun.earning_rate(2,current_slice[j_kai,1],current_slice[j,1],deal)
                    current_earn=fun.earning(2,current_slice[j_kai,1],current_slice[j,1],deal)
                    """
                    if prob>p1 and predictor==1:#开多仓交易,并且将时间轴后移time_width+1,直接进行下一步平仓工作
                        j_kai=j
                        j+=time_width#j+=time_width+1
                        for w in range(j_kai+1,j+time_width+1):#时刻更新KDJ值
                            paras[0],paras[1],_=fun.KDJ(current_slice[w-23:w+1,1],shoupan,paras)
                        if j>current_slice.shape[0]-1:
                            break
                        current_earn_rate=fun.earning_rate(1,current_slice[j_kai,1],current_slice[j,1],deal)
                        cur+=fun.earning_rate(1,current_slice[j_kai,1],current_slice[j,1],deal)
                        current_earn=fun.earning(1,current_slice[j_kai,1],current_slice[j,1],deal)
                        total_deal+=1
                        if current_earn_rate>0:#收益率为正
                            pos_curr_sum+=current_earn_rate
                            win+=1
                        else:
                            neg_curr_sum+=current_earn_rate
                            lose+=1
                    elif prob>p2 and predictor==2:#做空交易,并且将时间轴后移time_width+1,直接进行下一步平仓工作
                        j_kai=j
                        j+=time_width#j+=time_width+1
                        for w in range(j_kai+1,j+time_width+1):#时刻更新KDJ值
                            paras[0],paras[1],_=fun.KDJ(current_slice[w-23:w+1,1],shoupan,paras)
                        if j>current_slice.shape[0]-1:
                            break
                        current_earn_rate=fun.earning_rate(2,current_slice[j_kai,1],current_slice[j,1],deal)
                        cur+=fun.earning_rate(2,current_slice[j_kai,1],current_slice[j,1],deal)
                        current_earn=fun.earning(2,current_slice[j_kai,1],current_slice[j,1],deal)
                        total_deal+=1
                        if current_earn_rate>0:
                            pos_curr_sum+=current_earn_rate
                            win+=1
                        else:
                            neg_curr_sum+=current_earn_rate
                            lose+=1
                    else:
                        j+=1
                        continue
                    earning_rate_record.append(cur)
                    earnig.append(current_earn)
            j+=1
    return db,earning_rate_record,earnig,(win/total_deal),(pos_curr_sum/win),(neg_curr_sum/lose),abs((pos_curr_sum/win)/(neg_curr_sum/lose)),

        


if __name__ == '__main__':
    print("高频交易模拟")
    data=csv_file_read('data/T3_stock_management/601318.SH.xlsx.csv')
    data=np.delete(data,0,0)#删掉开头
    data=np.delete(data,0,1)
    data=data.astype("f8")
    #data=data[data[:,0]==117,:]
    print(data)
    db,earning_rate_record,earning_record,win_rate,avg_win,avg_lose,pos_neg=simulate(data)#收益率记录，单次收益记录，胜率，平均盈利，平均亏损，盈亏比
    plt.figure()
    plt.plot(np.arange(len(earning_rate_record)),np.array(earning_rate_record))
    plt.show()
    print('胜率为：{}%,平均盈利:{}，平均亏损:{}，盈亏比:{},总收益:{},三日最大回撤:{}'.format(win_rate,avg_win*1000,avg_lose*1000,pos_neg,sum(earning_record),db))
