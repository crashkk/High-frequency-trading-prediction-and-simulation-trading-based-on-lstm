from pandas import *#该py解决问题一的前三十个数据集指标提取问题
import os
import numpy as np
import scipy.io
import csv
from math import *
from feature_func import totalfunc
from collections import Counter
from dataprocess import *

for filename in os.listdir('data/new_data'):
    path='{}/{}'.format('data/new_data',filename)#数据提取与处理
    dataf=read_excel(path,sheet_name=0)
    dataf_np=np.array(dataf)
    dataf_np=np.delete(dataf_np,0,0)
    dataf_np=np.delete(dataf_np,0,1)
    jiaoyie=data_reconstruct(dataf_np,'data/csv_file/{}.csv'.format(filename))#重构股票数据，提取出平均价格、交易额等信息
    new_data=csv_file_read('data/csv_file/{}.csv'.format(filename))
    new_data=np.delete(new_data,0,0)
    new_data=new_data.astype("f8")
    st_day=0

    last_deal=[]#最后时刻的成交额
    wave_perday_max1=[]#每日突破最大值的次数
    wave_perday_min1=[]#每日跌落最小值的次数
    wave_perday_max2=[]#幅度
    wave_perday_min2=[]
    wave_k=[]#幅度指标

    new_wave=[]
    d=0#天数
    dd=[]
    for i in range(new_data.shape[0]):
        day=new_data[i,0]
        if day!=st_day:#更新天数
            d+=1
            dd.append(day)
            st_day=day
            slice=new_data[new_data[:,0]==st_day,:]
            #计算最后时刻的成交额
            last_deal.append(jiaoyie[str(int(st_day))][-1])
            max=slice[0,-2]
            min=slice[0,-1]
            M=0
            N=0
            wave_max=0
            wave_min=0

            max_price=np.max(slice[:,5])#提取当日最高价和最低价
            min_price=np.min(slice[:,6])
            shoupan=slice[0,3]#前一日收盘价
            new_wave.append(np.max(np.array([abs(max_price-min_price),abs(max_price-shoupan),abs(min_price-shoupan)])))#波动幅度
            for j in range(1,slice.shape[0]):#统计股价突破最大、小值的次数和幅度
                if slice[j,-2]>max:
                    max_b=max#保存次大值
                    max=slice[j,-2]
                    M+=1
                    wave_max+=abs(max-max_b)/max_b

                if slice[j,-1]<min:
                    min_b=min
                    min=slice[j,-1]
                    N+=1
                    wave_min+=abs(min-min_b)/min_b
            wave_perday_max1.append(M)
            wave_perday_min1.append(N)
            wave_perday_max2.append(wave_max)
            wave_perday_min2.append(wave_min)    

        else:
            continue
    
    print('股票：{}的突破最值平均次数：{}'.format(filename,(sum(wave_perday_max1)+sum(wave_perday_min1))/d))
    print('突破最值平均幅度：{}'.format((sum(wave_perday_max2)+sum(wave_perday_min2))/(sum(wave_perday_max1)+sum(wave_perday_min1)),))

    csv_file = open('data/csv_file_target/{}.csv'.format(filename), 'w', newline='', encoding='gbk')
    writer = csv.writer(csv_file)
    header=['日期','每日最后时刻的成交额','每日突破最大值的次数','每日跌落最小值的次数','每日总涨幅','每日总跌幅','波动幅度']
    writer.writerow(header)
    for i in range(len(wave_perday_max1)):
        writer.writerow([dd[i]]+[last_deal[i]]+[wave_perday_max1[i]]+[wave_perday_min1[i]]+[wave_perday_max2[i]]+[wave_perday_min2[i]]+[new_wave[i]])
    writer.writerow([(sum(wave_perday_max1)+sum(wave_perday_min1))/d]+[(sum(wave_perday_max2)+sum(wave_perday_min2))/(sum(wave_perday_max1)+sum(wave_perday_min1))]+[np.mean(np.array(new_wave))])
    csv_file.close()

    csv_file2=open('ziye','a+')
    swriter=csv.writer(csv_file2)
    swriter.writerow([(sum(wave_perday_max1)+sum(wave_perday_min1))/d]+[(sum(wave_perday_max2)+sum(wave_perday_min2))/(sum(wave_perday_max1)+sum(wave_perday_min1))]+[np.mean(np.array(new_wave))])
    csv_file2.close()
    print('file:{} save to csv success!'.format(filename))
        
