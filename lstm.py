
from pandas import *
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
import random
import matplotlib.pyplot as plt 
history=LossHistory()
time_steps=6#时间窗大小

def dataset_balance(data,label):#均衡数据集
    np.random.seed(666)#设置随机种子，确保每次处理剔除的样本一致
    dim=label.shape[0]
    count=Counter(list(label))
    if count[1]>count[2]:#样本1多于样本2，选择随机剔除样本1的部分样本
        balance_num=count[1]-count[2]
        select=1
    elif count[1]==count[2]:#数据集完全均衡，无需再分配
        return data,label
    else:
        balance_num=count[2]-count[1]
        select=2
    print('发现异常标签:{}'.format(count[0]))
    all_pos=np.where(label==select)
    all_pos=np.squeeze(all_pos)
    dele_pos=np.random.choice(all_pos,balance_num,replace=False)
    label=np.delete(label,dele_pos,axis=0)
    data=np.delete(data,dele_pos,axis=0)
    count=Counter(list(label))
    print('均衡后的样本规模:{}涨 {}跌'.format(count[1],count[2]))
    return data,label

#加载训练、预测数据集
data,label=create_dataset(time_steps,path='data/csv_file/002594.SZ.xlsx.csv',save_path='data_lstm/dataset002594data.npy',label_path='data_lstm/dataset002594label.npy')
data=np.array(data)
label=np.array(label)

data,label=dataset_balance(data,label)

labels=get_dummies(label).values

SAMPLE_NUMS=data.shape[0]#数据集大小
test_size_pro=0.1

train_data,test_data,train_labels,test_labels=train_test_split(data,labels,test_size=test_size_pro,random_state=22)#分割数据集

DIMENSION=data.shape[2]#数据集中每个样本的特征数
hidden_layers1=64#自定义lstm隐藏层大小
hidden_layers2=32
BATCH_SIZE=100
train_epoachs=150#训练轮数

#搭建lstm网络
print(train_data.shape,train_labels.shape)
model=Sequential()
model.add(LSTM(hidden_layers1,dropout=0.2,return_sequences=True))
model.add(LSTM(hidden_layers2,dropout=0.2,recurrent_dropout=0.2,return_sequences=False))
model.add(Dense(2,activation="softmax"))
model.add(Activation('softmax'))#交叉熵损失函数

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])#用adam自适应优化算法

#训练模型
model.fit(train_data,train_labels,batch_size=BATCH_SIZE,epochs=train_epoachs,validation_data=(test_data,test_labels),callbacks=[history])

#预测评估
pred=model.predict(test_data)
print(pred)
pred=pred.argmax(axis=1)
test_labels=test_labels.argmax(axis=1)
print('测试集正确率是：%s' %accuracy_score(pred,test_labels))
target_names = ['上涨','下跌']
print(classification_report(test_labels, pred, target_names=target_names))
history.loss_plot('epoch')
print('保存lstm')
model.save('model_save/002594model.h5')

