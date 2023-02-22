from pandas import *#模拟高频交易投资组合
from dataprocess import *
import os
import numpy as np
import scipy.io
import csv
import tensorflow as tf
from loss_history import LossHistory
from keras import backend as K
from feature_func import totalfunc
from collections import Counter
from keras.layers.core import Activation, Dense, SpatialDropout1D
from keras.layers import Embedding
from keras.layers import LSTM,Flatten,Dense
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataprocess import csv_file_read,time_extract
import random
import matplotlib.pyplot as plt 
from keras.models import load_model

import pyswarms as ps#调用PSO粒子群算法求解最优化问题
swarm = 50
dim = 4        # 四个权重
epsilon = 1.0
options = {'c1': 1.5, 'c2':1.5, 'w':0.5}

constraints = ([0,0,0,0],
               [1,1,1,1])

##########加载多个模型，同时建立多个session,避免graph冲突#############
seg_graph1 = tf.Graph()
sess1 = tf.compat.v1.Session(graph=seg_graph1)
K.set_session(sess1)
# 加载模型1
with sess1.as_default():
    with seg_graph1.as_default():
    	model1 = load_model("model_save/000333model.h5")

seg_graph2 = tf.Graph()
sess2 = tf.compat.v1.Session(graph=seg_graph2)
K.set_session(sess2)
# 加载模型2
with sess2.as_default():
    with seg_graph2.as_default():
    	model2 = load_model("model_save/300014model.h5")

seg_graph3 = tf.Graph()
sess3 = tf.compat.v1.Session(graph=seg_graph3)
K.set_session(sess3)
# 加载模型3
with sess3.as_default():
    with seg_graph3.as_default():
    	model3 = load_model("model_save/600323model.h5")

seg_graph4 = tf.Graph()
sess4 = tf.compat.v1.Session(graph=seg_graph4)
K.set_session(sess4)
# 加载模型4
with sess4.as_default():
    with seg_graph4.as_default():
    	model4 = load_model("model_save/601318model.h5")
##################################################################
#########################一些全局变量的定义########################
fun=totalfunc()
#ind=2#补充的收盘价
suppliment=[43.04,96.63,19.05,38.37]#补充数据(000333.SZ,300014.SZ,600323.SH,601318.SH)
                                    #11.7[43.12,97.49,19.18,38.79
                                    #11.9
lamb=0.5#权重分配中的厌恶系数
time_width=3
feature_num=17#样本特征数
P1=0.65#000333                                        #四个股票各自的置信度参数(best p=0.61 for000049;best p=0.675 time_step=3 for 300014;best p=0.72 time_step=3 for 601318)
P2=0.65#300014
P3=0.65#600323
P4=0.65#601318
deal=5#自定义每次的成交量
initial_stock=[0,0,0,0]#存储最开始就买有的市值股票股数
individual_property=np.array([0,0,0,0])#存储四个股票的实时资产
inherrent_property=[0,0,0,0]#存储固定资产
r1=r2=r3=r4=p1=p2=p3=p4=kaipan=0
##################################################################

def time_tohourmin_second(time):#输入列向量ndarray,用于将模糊的长序列转换为小时-分钟的数字序列
    time=(time%1e6)//100
    return time

def KDJ_cal(data,shoupan):#接收数据矩阵，返回同维的KDJ值，还应该带有当天的收盘价
    KDJ_mat=np.zeros((data.shape[0]-10,3))
    paras=[50,50]
    for i in range(10,data.shape[0]):
        KDJ_mat[i,1]=fun.KDJ(data[i-23:i+1,1],shoupan,paras)

def constraint(x):#等式约束条件
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    return [x1+x2+x3+x4-1]

def PSO_cost(x):#pso的代价函数
    global r1,r2,r3,r4,p1,p2,p3,p4,kaipan
    r1=np.array(r1)
    r2=np.array(r2)
    r3=np.array(r3)
    r4=np.array(r4)

    loss1=x[:,0]*np.mean(r1)+x[:,1]*np.mean(r2)+x[:,2]*np.mean(r3)+x[:,3]*np.mean(r4)

    jump1=np.zeros(len(p1))
    jump2=np.zeros(len(p2))
    jump3=np.zeros(len(p3))
    jump4=np.zeros(len(p4))
    k=0.04
    for i in range(1,len(p1)):
        if (p1[i]/kaipan)>=k:
            jump1[i]=1
        else:
            jump1[i]=0
    for i in range(1,len(p2)):
        if (p2[i]/kaipan)>=k:
            jump2[i]=1
        else:
            jump2[i]=0
    for i in range(1,len(p2)):
        if (p2[i]/kaipan)>=k:
            jump2[i]=1
        else:
            jump2[i]=0
    for i in range(1,len(p2)):
        if (p2[i]/kaipan)>=k:
            jump2[i]=1
        else:
            jump2[i]=0
    loss2=pow(x[:,0],2)*fun.v(r1)+pow(x[:,1],2)*fun.v(r2)+pow(x[:,2],2)*fun.v(r3)+pow(x[:,3],2)*fun.v(r4)
    cov_12=0
    for i in range(len(p1)):
            for j in range(len(p2)):
                cov_12+=r1[i]*r2[j]*jump1[i]*jump2[j]
    cov_13=0
    for i in range(len(p1)):
            for j in range(len(p3)):
                cov_13+=r1[i]*r3[j]*jump1[i]*jump3[j]
    cov_23=0
    for i in range(len(p2)):
            for j in range(len(p3)):
                cov_23+=r2[i]*r3[j]*jump2[i]*jump3[j]
    cov_14=0
    for i in range(len(p1)):
            for j in range(len(p4)):
                cov_14+=r1[i]*r4[j]*jump1[i]*jump4[j]
    cov_34=0
    for i in range(len(p3)):
            for j in range(len(p4)):
                cov_34+=r3[i]*r4[j]*jump3[i]*jump4[j]
    loss2+=x[:,0]*x[:,1]*cov_12+x[:,0]*x[:,2]*cov_13+x[:,1]*x[:,2]*cov_23+x[:,2]*x[:,3]*cov_34+x[:,0]*x[:,3]*cov_14
    return -(1-lamb)*loss1-lamb*loss2+1-x[:,0]+x[:,1]+x[:,2]+x[:,3]
def weight_judgement(weight,R1,R2,R3,R4,P1,P2,P3,P4,kp):#权重更新，细致的数学运算。接收参数为待更新的权重，以及四个股票的这一分钟的收益率总结,四个list,和当天开盘价
    global r1,r2,r3,r4,p1,p2,p3,p4,kaipan
    r1=R1
    r2=R2
    r3=R3
    r4=R4
    p1=P1
    p2=P2
    p3=P3
    p4=P4
    kaipan=kp
    #if True:
    #    return weight
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm,
                                    dimensions=dim,
                                    options=options,
                                    bounds=constraints)
    cost, joint_vars = optimizer.optimize(PSO_cost, iters=100)
    joint_vars=np.array(joint_vars)
    joint_vars=fun.weight_reconstruct(joint_vars)
    print(joint_vars)
    return joint_vars
    """
    B=np.array([[2*fun.v(r1),fun.cov(r1,r2,p1,p2,kaipan),fun.cov(r1,r3,p1,p3,kaipan),fun.cov(r1,r4,p1,p4,kaipan),1],
                [fun.cov(r2,r1,p2,p1,kaipan),2*fun.v(r2),fun.cov(r2,r3,p2,p3,kaipan),fun.cov(r2,r4,p2,p4,kaipan),1],
                [fun.cov(r3,r1,p3,p1,kaipan),fun.cov(r3,r2,p3,p2,kaipan),2*fun.v(r3),fun.cov(r3,r4,p3,p4,kaipan),1],
                [fun.cov(r4,r1,p4,p1,kaipan),fun.cov(r4,r2,p4,p2,kaipan),fun.cov(r4,r3,p4,p3,kaipan),2*fun.v(r4),1]])
    C=np.array([fun.E(r1),fun.E(r2),fun.E(r3),fun.E(r4),lamb/(1-lamb)])
    B=np.linalg.inv(B)#求逆
    X=((1-lamb)/lamb)*C*B
    print('当前权重:{}'.format(X[:,4]))
    return X[:4]
    """

def predict(pred_data,model_num):#模型给出预测。接收参数必须为ndarry的形式,1*time_width*features_num的一个三维张量。最后一个接收参数输入模型序号
    if model_num==1:
        with sess1.as_default():
            with seg_graph1.as_default():
                pred=model1.predict(np.expand_dims(pred_data,axis=0))
    elif model_num==2:
        with sess2.as_default():
            with seg_graph2.as_default():
                pred=model2.predict(np.expand_dims(pred_data,axis=0))
    elif model_num==3:
        with sess3.as_default():
            with seg_graph3.as_default():
                pred=model3.predict(np.expand_dims(pred_data,axis=0))
    elif model_num==4:
        with sess4.as_default():
            with seg_graph4.as_default():
                pred=model4.predict(np.expand_dims(pred_data,axis=0))

    pred=np.squeeze(pred)
    pred=pred.tolist()
    
    prob=max(pred[0],pred[1])
    hint=pred.index(prob)+1#1:上涨,2:下跌
    return hint,prob#发生概率最大事件提示，和其概率

def one_by_one_simulate_permin(p,data,start_second,shoupan,KDJ,model_num):#该模块用于构造预测矩阵，并且完成一分钟内的模拟交易。接收参数为该股票的置信度超参数,这一分钟加上前一分钟一部分的所有tick构成的数据矩阵,
                                                                     #以及该分钟第一秒在矩阵中的位置,还有当天的收盘价,该分钟内的KDJ,模型序号
    global individual_property,initial_stock,inherrent_property
    wait_signal=0
    time_cal=0
    start_second=start_second+10#在第11秒才选择开始
    pred_matrix=np.zeros((time_width,feature_num))
    j=start_second
    earn_rate=[]#存储每次收益率
    earning=[]#存储每次的收益情况（盈利或亏本）
    price=[]#存储出手时相应的股价
    while j<=data.shape[0]-1:
        if wait_signal==0 and time_cal<=time_width-1:#情况一，前一秒发生了空仓，因此数据要从0到time_width重新刷新
            jia_ma_6=fun.jiage_MA(data[j-23:j+1,1],6)#增加新特征
            jia_ma_12=fun.jiage_MA(data[j-23:j+1,1],12)
            jia_ma_24=fun.jiage_MA(data[j-23:j+1,1],24)
            cjl_ma_6=fun.chengjiaoliang_MA(data[j-23:j+1,4],6)
            cjl_ma_12=fun.chengjiaoliang_MA(data[j-23:j+1,4],12)
            cjl_ma_24=fun.chengjiaoliang_MA(data[j-23:j+1,4],24)
            pclimb=fun.p_climb_rate(data[j-23:j+1,1])
            cclimb=fun.c_climb_rate(data[j-23:j+1,4])
            
            pred_matrix[time_cal,:]=np.array([data[j,1],data[j,4],data[j,5],data[j,6],#为该时刻写入特征
                jia_ma_6,jia_ma_12,jia_ma_24,cjl_ma_6,cjl_ma_12,cjl_ma_24,pclimb,cclimb,KDJ[j,0],KDJ[j,1],KDJ[j,2],data[j,7],data[j,8]])
            time_cal+=1
        if wait_signal==1:
            jia_ma_6=fun.jiage_MA(data[j-23:j+1,1],6)#增加新特征
            jia_ma_12=fun.jiage_MA(data[j-23:j+1,1],12)
            jia_ma_24=fun.jiage_MA(data[j-23:j+1,1],24)
            cjl_ma_6=fun.chengjiaoliang_MA(data[j-23:j+1,4],6)
            cjl_ma_12=fun.chengjiaoliang_MA(data[j-23:j+1,4],12)
            cjl_ma_24=fun.chengjiaoliang_MA(data[j-23:j+1,4],24)
            pclimb=fun.p_climb_rate(data[j-23:j+1,1])
            cclimb=fun.c_climb_rate(data[j-23:j+1,4])
            pred_matrix=np.delete(pred_matrix,0,axis=0)#第一行数据删除，用当前时刻的插入到最后
            insertone=np.array([data[j,1],data[j,4],data[j,5],data[j,6],#为该时刻写入特征
                jia_ma_6,jia_ma_12,jia_ma_24,cjl_ma_6,cjl_ma_12,cjl_ma_24,pclimb,cclimb,KDJ[j,0],KDJ[j,1],KDJ[j,2],data[j,7],data[j,8]])
            pred_matrix=np.concatenate([pred_matrix,insertone],axis=0)
        if time_cal==time_width:
            pred_matrix=fun.normalization(pred_matrix)
            wait_signal=0
            predictor,prob=predict(pred_matrix,model_num)
            if prob>p and predictor==1:#预测上涨，选择做多,然后平仓
                wait_signal=0
                j_kai=j
                j+=time_width
                if(len(earn_rate)==0):#该分钟内的第一次出手，若是做多，则将市值资金全部卖掉变现
                    individual_property[model_num-1]+=initial_stock[model_num-1]*data[j,1]*100
                earn_rate.append(fun.earning_rate(1,data[j_kai,1],data[j,1],individual_property[model_num-1]/data[j,1]/100))#计算收益率。每次能投入的钱是当前时刻所有的资金
                earning.append(fun.earning(1,data[j_kai,1],data[j,1],individual_property[model_num-1]/data[j,1]/100))
                price.append(abs(data[j,1]-data[j-1,1]))
                individual_property[model_num-1]+=fun.earning(1,data[j_kai,1],data[j,1],individual_property[model_num-1]/data[j,1]/100)
                time_cal=0
            elif prob>p and predictor==2:#预测下跌，选择做空,然后平仓
                wait_signal=0
                j_kai=j
                j+=time_width
                if(len(earn_rate)==0):#该分钟内的第一次出手，若是做空，则在做空决定的那一秒将市值资金全部卖掉变现
                    individual_property[model_num-1]+=initial_stock[model_num-1]*data[j_kai,1]*100
                earn_rate.append(fun.earning_rate(2,data[j_kai,1],data[j,1],individual_property[model_num-1]/data[j_kai,1]/100))#计算收益率。每次能投入的钱是当前时刻所有的资金
                earning.append(fun.earning(2,data[j_kai,1],data[j,1],individual_property[model_num-1]/data[j_kai,1]/100))
                price.append(abs(data[j,1]-data[j-1,1]))
                individual_property[model_num-1]+=fun.earning(2,data[j_kai,1],data[j,1],individual_property[model_num-1]/data[j_kai,1]/100)
                time_cal=0
            else:#不做行动，等待下一秒数据更新
                wait_signal==1#指示下一次只需要更新数据矩阵最后一个tick即可
                j+=1
                if j+time_width>data.shape[0]-1:
                    if len(earn_rate)>0:
                        individual_property[model_num-1]=individual_property[model_num-1]-initial_stock[model_num-1]*100*data[j,1]
                        inherrent_property[model_num-1]=initial_stock[model_num-1]*100*data[j,1]
                    else:
                        earn_rate=[0]#不出手，收益率为0
                        earning=[0]
                    break
                continue
        j+=1
        #此部分用来判断该分钟的高频交易是否完全结束，并且将原市值的股数在当前时刻全部买回去，总资金要相应地剪去那部分钱,余下的钱就是流动资金的目前状态
        if j+time_width>data.shape[0]-1 or j>=data.shape[0]-1 :
            if len(earn_rate)>0:
                if j>data.shape[0]-1:
                    individual_property[model_num-1]=individual_property[model_num-1]-initial_stock[model_num-1]*100*data[data.shape[0]-1,1]
                    inherrent_property[model_num-1]=initial_stock[model_num-1]*100*data[data.shape[0]-1,1]
                else:
                    individual_property[model_num-1]=individual_property[model_num-1]-initial_stock[model_num-1]*100*data[j,1]
                    inherrent_property[model_num-1]=initial_stock[model_num-1]*100*data[data.shape[0]-1,1]
            else:
                earn_rate=[0]
                earning=[0]
            break
    
    return earn_rate,earning,price#返回该股票在这一分钟每次出手的收益率和净收益列表和出手股价之差



def simulate_multiple_invest(data1,data2,data3,data4):#模拟股票组合交易,提取出四个股票数据。开头索引和第一列应该已经删除掉。接收该股票所有tick数据
    global individual_property,initial_stock,inherrent_property
    data1=data1.astype("f8")
    data2=data2.astype("f8")
    data3=data3.astype("f8")
    data4=data4.astype("f8")
    daylist1=data1[:,0]
    daylist1=set(list(daylist1))#股票1天数集合,存有所有的天数信息(1011,1012....)
    daylist1=list(daylist1)
    daylist1.sort()#将天数从小到大排序
    dd=0
    totalearn=[]
    inherrent=[]#查看固定资产的变动情况
    totalearn_rate=[]
    weight_rd=[]
    earning_rate_stock=[]
    max_drawback=[]#记录四个股票每日最大回撤
    for d in daylist1:#所有股票共有时间段(9:30~11:30am,13:00~15:00,不包含11:30和15:00这一分钟）
        if d==119:
            shoupan1=suppliment[0]
            shoupan2=suppliment[1]
            shoupan3=suppliment[2]
            shoupan4=suppliment[3]
        else:
            shoupan1=data1[data1[:,0]==daylist1[dd+1],3]
            shoupan1=shoupan1[0]
            shoupan2=data2[data2[:,0]==daylist1[dd+1],3]
            shoupan2=shoupan2[0]
            shoupan3=data3[data3[:,0]==daylist1[dd+1],3]
            shoupan3=shoupan3[0]
            shoupan4=data4[data4[:,0]==daylist1[dd+1],3]
            shoupan4=shoupan4[0]
        print(shoupan1)
        
        current_slice1=data1[data1[:,0]==d,:]#股票1某一天的所有数据
        current_slice2=data2[data2[:,0]==d,:]#股票2某一天的所有数据
        current_slice3=data3[data3[:,0]==d,:]#股票3某一天的所有数据
        current_slice4=data4[data4[:,0]==d,:]#股票4某一天的所有数据
        current_slice1[:,-1]=time_tohourmin_second(current_slice1[:,-1])
        current_slice2[:,-1]=time_tohourmin_second(current_slice2[:,-1])
        current_slice3[:,-1]=time_tohourmin_second(current_slice3[:,-1])
        current_slice4[:,-1]=time_tohourmin_second(current_slice4[:,-1])
        
        KDJ1=np.zeros((current_slice1.shape[0],3))
        KDJ2=np.zeros((current_slice2.shape[0],3))
        KDJ3=np.zeros((current_slice3.shape[0],3))
        KDJ4=np.zeros((current_slice4.shape[0],3))
        paras=[50,50]
        for q in range(23,KDJ1.shape[0]):
            KDJ1[q,0],KDJ1[q,1],KDJ1[q,2]=fun.KDJ(current_slice1[q-23:q+1,1],shoupan1,paras)
            paras=[KDJ1[q,0],KDJ1[q,1]]
        paras=[50,50]
        for q in range(23,KDJ2.shape[0]):
            KDJ2[q,0],KDJ2[q,1],KDJ2[q,2]=fun.KDJ(current_slice2[q-23:q+1,1],shoupan2,paras)
            paras=[KDJ2[q,0],KDJ2[q,1]]
        paras=[50,50]
        for q in range(23,KDJ3.shape[0]):
            KDJ3[q,0],KDJ3[q,1],KDJ3[q,2]=fun.KDJ(current_slice3[q-23:q+1,1],shoupan3,paras)
            paras=[KDJ3[q,0],KDJ3[q,1]]
        paras=[50,50]
        for q in range(23,KDJ4.shape[0]):
            KDJ4[q,0],KDJ4[q,1],KDJ4[q,2]=fun.KDJ(current_slice4[q-23:q+1,1],shoupan4,paras)
            paras=[KDJ4[q,0],KDJ4[q,1]]
        #####初始化######
        minlist=set(list(current_slice1[:,-1]))
        minlist=list(minlist)
        minlist.sort()#存储好分钟序列(932,933,934...)
        start_time=932#从9点32分开始高频交易
        current_min1=np.concatenate([current_slice1[current_slice1[:,-1]==start_time-1,:],current_slice1[current_slice1[:,-1]==start_time,:]],axis=0)#提取股票1处当前分钟的所有行情序列
        current_min2=np.concatenate([current_slice2[current_slice2[:,-1]==start_time-1,:],current_slice2[current_slice2[:,-1]==start_time,:]],axis=0)#提取股票2处当前分钟的所有行情序列
        current_min3=np.concatenate([current_slice3[current_slice3[:,-1]==start_time-1,:],current_slice3[current_slice3[:,-1]==start_time,:]],axis=0)#提取股票3处当前分钟的所有行情序列
        current_min4=np.concatenate([current_slice4[current_slice4[:,-1]==start_time-1,:],current_slice4[current_slice4[:,-1]==start_time,:]],axis=0)#提取股票4处当前分钟的所有行情序列
        KDJ1_current_min1=np.concatenate([KDJ1[current_slice1[:,-1]==start_time-1,:],KDJ1[current_slice1[:,-1]==start_time,:]],axis=0)#提取股票1处当前分钟的所有KDJ值
        KDJ2_current_min2=np.concatenate([KDJ2[current_slice2[:,-1]==start_time-1,:],KDJ2[current_slice2[:,-1]==start_time,:]],axis=0)#提取股票2处当前分钟的所有KDJ值
        KDJ3_current_min3=np.concatenate([KDJ3[current_slice3[:,-1]==start_time-1,:],KDJ3[current_slice3[:,-1]==start_time,:]],axis=0)#提取股票3处当前分钟的所有KDJ值
        KDJ4_current_min4=np.concatenate([KDJ4[current_slice4[:,-1]==start_time-1,:],KDJ4[current_slice4[:,-1]==start_time,:]],axis=0)#提取股票4处当前分钟的所有KDJ值
        initial_stock[0]=(1500000/current_min1[0,3])/100#按手来算
        initial_stock[1]=(1500000/current_min2[0,3])/100
        initial_stock[2]=(1500000/current_min3[0,3])/100
        initial_stock[3]=(1500000/current_min4[0,3])/100
        #################
        for i in range(238):#一天四个小时,238分钟
            if i==0:
                if d==117:
                    weight=np.array([0.25,0.25,0.25,0.25])#给第一天的流动资金分配权重
                    individual_property=individual_property+4000000*weight#注入一定比例流动资金
                earn_rate1,earn1,price1=one_by_one_simulate_permin(P1,current_min1,current_slice1[current_slice1[:,-1]==start_time-1,:].shape[0]-1,shoupan1,KDJ1_current_min1,1)
                earn_rate2,earn2,price2=one_by_one_simulate_permin(P2,current_min2,current_slice2[current_slice2[:,-1]==start_time-1,:].shape[0]-1,shoupan2,KDJ2_current_min2,2)
                earn_rate3,earn3,price3=one_by_one_simulate_permin(P3,current_min3,current_slice3[current_slice3[:,-1]==start_time-1,:].shape[0]-1,shoupan3,KDJ3_current_min3,3)
                earn_rate4,earn4,price4=one_by_one_simulate_permin(P4,current_min4,current_slice4[current_slice4[:,-1]==start_time-1,:].shape[0]-1,shoupan4,KDJ4_current_min4,4)
                continue
            else:
                earn_rate1,earn1,price1=one_by_one_simulate_permin(P1,current_min1,current_slice1[current_slice1[:,-1]==start_time-1,:].shape[0]-1,shoupan1,KDJ1_current_min1,1)
                earn_rate2,earn2,price2=one_by_one_simulate_permin(P2,current_min2,current_slice2[current_slice2[:,-1]==start_time-1,:].shape[0]-1,shoupan2,KDJ2_current_min2,2)
                earn_rate3,earn3,price3=one_by_one_simulate_permin(P3,current_min3,current_slice3[current_slice3[:,-1]==start_time-1,:].shape[0]-1,shoupan3,KDJ3_current_min3,3)
                earn_rate4,earn4,price4=one_by_one_simulate_permin(P4,current_min4,current_slice4[current_slice4[:,-1]==start_time-1,:].shape[0]-1,shoupan4,KDJ4_current_min4,4)
            
            print('stock1{}'.format(earn_rate1))
            print('stock2{}'.format(earn_rate2))
            print('stock3{}'.format(earn_rate3))
            print('stock4{}'.format(earn_rate4))
            #print(individual_property)
            earning_rate_stock.append([np.mean(np.array(earn_rate1)),np.mean(np.array(earn_rate2)),np.mean(np.array(earn_rate3)),np.mean(np.array(earn_rate4))])
            totalearn.append([individual_property[0],individual_property[1],individual_property[2],individual_property[3]])
            inherrent.append(np.sum(inherrent_property))
            weight_rd.append(list(weight))
            #totalearn_rate.append()
            if len(earn_rate1)!=0 and len(earn_rate2)!=0 and len(earn_rate3)!=0 and len(earn_rate4)!=0:
                weight=weight_judgement(weight,earn_rate1,earn_rate2,earn_rate3,earn_rate4,price1,price2,price3,price4,current_slice1[0,2])
            individual_property=sum(individual_property)*weight
        ###########################重新初始化###########################        
            current_min1=np.concatenate([current_slice1[current_slice1[:,-1]==minlist[i+1],:],current_slice1[current_slice1[:,-1]==minlist[i+2],:]],axis=0)#提取股票1处当前分钟的所有行情序列
            current_min2=np.concatenate([current_slice2[current_slice2[:,-1]==minlist[i+1],:],current_slice2[current_slice2[:,-1]==minlist[i+2],:]],axis=0)#提取股票2处当前分钟的所有行情序列
            current_min3=np.concatenate([current_slice3[current_slice3[:,-1]==minlist[i+1],:],current_slice3[current_slice3[:,-1]==minlist[i+2],:]],axis=0)#提取股票3处当前分钟的所有行情序列
            current_min4=np.concatenate([current_slice4[current_slice4[:,-1]==minlist[i+1],:],current_slice4[current_slice4[:,-1]==minlist[i+2],:]],axis=0)#提取股票4处当前分钟的所有行情序列
            KDJ1_current_min1=np.concatenate([KDJ1[current_slice1[:,-1]==minlist[i+1],:],KDJ1[current_slice1[:,-1]==minlist[i+2],:]],axis=0)#提取股票1处当前分钟的所有KDJ值
            KDJ2_current_min2=np.concatenate([KDJ2[current_slice2[:,-1]==minlist[i+1],:],KDJ2[current_slice2[:,-1]==minlist[i+2],:]],axis=0)#提取股票2处当前分钟的所有KDJ值
            KDJ3_current_min3=np.concatenate([KDJ3[current_slice3[:,-1]==minlist[i+1],:],KDJ3[current_slice3[:,-1]==minlist[i+2],:]],axis=0)#提取股票3处当前分钟的所有KDJ值
            KDJ4_current_min4=np.concatenate([KDJ4[current_slice4[:,-1]==minlist[i+1],:],KDJ4[current_slice4[:,-1]==minlist[i+2],:]],axis=0)#提取股票4处当前分钟的所有KDJ值


        dd+=1
        max_drawback.append([drawback(current_slice1[:,1]),drawback(current_slice2[:,1]),drawback(current_slice3[:,1]),drawback(current_slice4[:,1])])

    
    totalearn=np.array(totalearn)#打印每个股票收益变化
    print(totalearn)
    plt.figure()
    plt.plot(np.arange(totalearn.shape[0]),totalearn[:,0],color='red')
    plt.plot(np.arange(totalearn.shape[0]),totalearn[:,1],color='blue')
    plt.plot(np.arange(totalearn.shape[0]),totalearn[:,2],color='green')
    plt.plot(np.arange(totalearn.shape[0]),totalearn[:,3],color='yellow')
    plt.xlabel('time-axis')
    plt.ylabel('earn')
    plt.legend(['000333.SZ','300014.SZ','600323.SH','601318.SH'])
    plt.show()

    plt.figure()
    plt.plot(np.arange(totalearn.shape[0]),np.sum(totalearn,axis=1),color='blue')
    plt.xlabel('time-axis')
    plt.ylabel('totalearn')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(inherrent)),np.array(inherrent),color='red')#固定资产的变化情况
    plt.xlabel('time-axis')
    plt.ylabel('inherent')
    plt.show()

    earning_rate_stock=np.array(earning_rate_stock)#11.7-11.9平均收益率变化图像
    plt.figure()
    plt.plot(np.arange(earning_rate_stock.shape[0]),earning_rate_stock[:,0],color='red')
    plt.plot(np.arange(earning_rate_stock.shape[0]),earning_rate_stock[:,1],color='blue')
    plt.plot(np.arange(earning_rate_stock.shape[0]),earning_rate_stock[:,2],color='green')
    plt.plot(np.arange(earning_rate_stock.shape[0]),earning_rate_stock[:,3],color='yellow')
    plt.xlabel('time-axis')
    plt.ylabel('earning_rate')
    plt.legend(['000333.SZ','300014.SZ','600323.SH','601318.SH'])
    plt.show()

    weight_rd=np.array(weight_rd)#11.7-11.9平均收益率变化图像
    plt.figure()
    plt.plot(np.arange(weight_rd.shape[0]),weight_rd[:,0],color='red')
    plt.plot(np.arange(weight_rd.shape[0]),weight_rd[:,1],color='blue')
    plt.plot(np.arange(weight_rd.shape[0]),weight_rd[:,2],color='green')
    plt.plot(np.arange(weight_rd.shape[0]),weight_rd[:,3],color='yellow')
    plt.xlabel('time-axis')
    plt.ylabel('weight')
    plt.legend(['000333.SZ','300014.SZ','600323.SH','601318.SH'])
    plt.show()

    print('最终盈利:{}'.format(np.sum(totalearn[-1,:])+inherrent[-1]-10000000))
    print('最终流动资金分配:000333.SZ--{},300014.SZ--{},600323.SH--{},601318.SH--{}'.format(totalearn[-1,0],totalearn[-1,1],totalearn[-1,2],totalearn[-1,3]))
    print('每日最大回撤值:{}'.format(max_drawback))

def drawback(price):#计算每日最大回撤
    db=-1
    for i in range(price.shape[0]-1):
        for j in range(i+1,price.shape[0]):
            if price[j]-price[i]<0:
                if db<abs(price[j]-price[i])/price[i]:
                    db=abs(price[j]-price[i])/price[i]
            else: 
                break
    return db

if __name__ == '__main__':
    #select=117

    print("高频交易模拟投资组合")
    data1=csv_file_read('data/T3_stock_management/000333.SZ.xlsx.csv')
    data2=csv_file_read('data/T3_stock_management/300014.SZ.xlsx.csv')
    data3=csv_file_read('data/T3_stock_management/600323.SH.xlsx.csv')
    data4=csv_file_read('data/T3_stock_management/601318.SH.xlsx.csv')
    
    data1=np.delete(data1,0,0)#删掉开头
    data1=np.delete(data1,0,1)
    data1=data1.astype("f8")
    #data1=data1[data1[:,0]==select,:]
    
    data2=np.delete(data2,0,0)#删掉开头
    data2=np.delete(data2,0,1)
    data2=data2.astype("f8")
    #data2=data2[data2[:,0]==select,:]
    
    data3=np.delete(data3,0,0)#删掉开头
    data3=np.delete(data3,0,1)
    data3=data3.astype("f8")
    #data3=data3[data3[:,0]==select,:]
    
    data4=np.delete(data4,0,0)#删掉开头
    data4=np.delete(data4,0,1)
    data4=data4.astype("f8")
    #data4=data4[data4[:,0]==select,:]
    simulate_multiple_invest(data1,data2,data3,data4)
