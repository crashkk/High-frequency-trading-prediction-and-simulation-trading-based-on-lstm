from pandas import *
import os
import numpy as np
import scipy.io
import csv
from feature_func import totalfunc
from collections import Counter
import matplotlib.pyplot as plt 
ind=4#补充的收盘价
suppliment=[40.18,83.04,17.92,36.15,263]#补充数据(000333.SZ,300014.SZ,600323.SH,601318.SH,002594.SZ（比亚迪）)

def file_save_mat(data,name):#将ndarray数据保存为.mat
    scipy.io.savemat(name,mdict={'avg_price':data})
    print('file save to mat success!')
def file_save_csv(data,shoupan,kaipan,down,ceil,floor,name):#将ndarray数据保存为.csv
                                            #接收参数皆为字典数据(平均价，收盘价，开盘价,成交量,最高价，最低价)
    #data1 = DataFrame(data) # header:原第一行的索引，index:原第一列的索引
    #data1=data1.T
    #data1.to_csv(name)
    csv_file = open(name, 'w', newline='', encoding='gbk')
    writer = csv.writer(csv_file)
    head_row=['time','avg_price','kaipan','shoupan','down','ceil','floor']#具体时间(x月x日,数据一般从9:30开始)，平均股价，开盘价，收盘价，成交量
    writer.writerow(head_row)
    for key,value in data.items():
        for i in range(len(value)):
            writer.writerow([key]+[value[i]]+[shoupan[key]]+[kaipan[key]]+[down[key][i]]+[ceil[key][i]]+[floor[key][i]])
    
    csv_file.close()
    print('file save to csv success!')
def csv_file_read(name):#读取csv文件，并以ndarray的形式返回数据
    csv_file = open(name, 'r', newline='', encoding='gbk')#编码方式与写入文件时相同
    reader = csv.reader(csv_file)
    new_data=[]
    for row in reader:
        new_data.append(row)
    new_data=np.array(new_data)
    print('file read success')
    return new_data

def time_extract(time):#从源数据提取时间信息
    month=(time//(1e8))%100
    day=(time//1e6)%100
    hour=(time//1e4)%100
    min=(time//1e2)%100
    second=time%100
    return month,day,hour,min,second

def features_append(origin_path,file_name):#用于给数据集添加新的特征。接收new_data里的excell文件名，以及生成重构后存放csv文件的地址
    dataf=read_excel(origin_path,sheet_name=0)
    dataf_np=np.array(dataf)
    dataf_np=np.delete(dataf_np,0,0)
    dataf_np=np.delete(dataf_np,0,1)
    jiaoyie,time_list=data_reconstruct(dataf_np,file_name)

    data= read_csv(file_name,low_memory=False)
    print(len(time_list))
    print(data)
    b1=data['avg_price']
    b1=b1.diff()#求一阶差分
    b2=b1.diff()#求二阶差分
    plt.plot(b1)
    plt.show()
    b1=list(b1)
    b2=list(b2)
    data['diff1']=b1
    data['diff2']=b2
    data['exact_time']=time_list
    data=data.fillna(0)
    data.to_csv(file_name)
    print('new features add success')

def create_dataset(time_width,path,save_path,label_path):#为lstm构造数据集,path存储处理好的csv文件,time_width为输入lstm的样本固定时间间隔
    fun=totalfunc()
    new_data=csv_file_read(path)                           #[-----23------n----2*time_width---]
    #对于每个day而言，假设从t=1开始，t=end结束，起始可作为样本的tick应该从t=24开始选取，因为前t=23不存在MA(24)。
    #同样的，结尾可作为样本的tick应该在t=end-12结束选取，因为t=end+1处不存在标签
    #注意，这里的时间不是以秒做单位，定义1t=3sec
    header=new_data[0,:]#header存储csv头部信息
    new_data=np.delete(new_data,0,0)
    new_data=np.delete(new_data,0,1)

    new_data=new_data.astype("f8")
    start_pos=new_data[0,0]#初始化位置指针
    current_slice=new_data[new_data[:,0]==start_pos,:]
    sig=-1
    sample=[]
    label=[]
    total_samp=np.zeros((10,10))
    samp=np.zeros((10,10))
    count=0
    l=set(list(new_data[:,0]))
    l=list(l)
    l.sort()
    shoupan=new_data[new_data[:,0]==l[1],:]
    shoupan=shoupan[0,3]
    k=0
    print(shoupan)
    for i in range(new_data.shape[0]):#每遍循环创造一个样本数据.维度：（time_width*features_num==15）
        if new_data[i,0]==new_data[-1,0]:#最后一天的数据留做高频交易模拟
            break
        if new_data[i,0]!=start_pos:
            k+=1
            if k==len(l)-1:
                shoupan=suppliment[ind]
            else:
                shoupan=new_data[new_data[:,0]==l[k+1],:]
                shoupan=shoupan[0,3]
            sig+=current_slice.shape[0]
            start_pos=new_data[i,0]
            current_slice=new_data[new_data[:,0]==start_pos,:]
        end=current_slice.shape[0]#定义改日的起始tick位置
        current=i-sig-1#当前所在tick
        if current>=23 and current<=end-time_width-1:#满足这个范围的才可以创建样本
            if current==23:
                paras=[50,50]
            slice=current_slice[current-23:current+1,:]
            slice_pre=current_slice[current:current+2*time_width,:]

            jia_ma_6=fun.jiage_MA(slice[:,1],6)#增加新特征
            jia_ma_12=fun.jiage_MA(slice[:,1],12)
            jia_ma_24=fun.jiage_MA(slice[:,1],24)
            cjl_ma_6=fun.chengjiaoliang_MA(slice[:,4],6)
            cjl_ma_12=fun.chengjiaoliang_MA(slice[:,4],12)
            cjl_ma_24=fun.chengjiaoliang_MA(slice[:,4],24)
            pclimb=fun.p_climb_rate(slice[:,1])
            cclimb=fun.c_climb_rate(slice[:,4])
            K,D,J=fun.KDJ(slice[:,1],shoupan,paras)
            paras=[K,D]
            #print(slice_pre.shape)
            if ((slice_pre[-1,1]-slice_pre[time_width-1,1])/slice_pre[time_width-1,1])*1000>2:#涨幅
                tag=1#上涨
            else:
                tag=2#下跌
            if current<=end-1-2*time_width:#后13天都可以计算features,但无法加标签
                sample.append([slice[-1,1],slice[-1,4],slice[-1,5],slice[-1,6],jia_ma_6,jia_ma_12,jia_ma_24,cjl_ma_6,cjl_ma_12,cjl_ma_24,pclimb,cclimb,K,D,J,slice[-1,7],slice[-1,8],tag])
            else:
                sample.append([slice[-1,1],slice[-1,4],slice[-1,5],slice[-1,6],jia_ma_6,jia_ma_12,jia_ma_24,cjl_ma_6,cjl_ma_12,cjl_ma_24,pclimb,cclimb,K,D,J,slice[-1,7],slice[-1,8],0])
            
            if current==end-time_width-1:
                count+=1
                samples_day=np.array([sample[0]])
                for q in range(1,len(sample)):
                    samples_day=np.concatenate((samples_day,np.array([sample[q]])),axis=0)
                stand=samples_day[0:time_width,:-1]
                #标准化和归一化
                stand=fun.normalization(stand)
                total_samp=[stand]
                if count==1:
                    samp=total_samp
                    label.append(samples_day[0,-1])
                label.append(samples_day[0,-1])
                for j in range(1,samples_day.shape[0]-time_width-1):
                    stand2=samples_day[j:j+time_width,:-1]
                    #标准化和归一化
                    stand2=fun.normalization(stand2)
                    total_samp=np.concatenate((total_samp,[stand2]),axis=0)
                    label.append(samples_day[j,-1])
                samp=np.concatenate((samp,total_samp),axis=0)
                sample=[]
    c=Counter(label)
    print('train and test data process down.{}total and {}label1 {}label2'.format(samp.shape[0],c[1],c[2]))
    
    #将数据保存为npy
    np.save(save_path,samp)
    np.save(label_path,label)
    print(samp)
    return samp,label                
#特征集构造[平均价，成交量，最高价，最低价，ma指数（6个）,涨速（2个）,KDJ指标（三个）,标签]
            
        

    

def data_reconstruct(data,path):#股票数据重构，参数接受值为ndarray。
                        #该函数用于提取股票的平均价格等潜在信息，并保存为csv格式
    start_m,start_d,x,y,z=time_extract(float(data[0,0]))
    price_record=[]#记录平均价格
    price_record_day={}#记录每日的平均价格变化
    down_record_day={}#记录每日成交量变化
    kaipan_rd={}#记录每日的开盘价
    shoupan_rd={}#记录每日的收盘价
    ceil_rd={}#记录每日最高价波动
    floor_rd={}#记录每日最低价波动
    temp_ap=[]
    temp_dp=[]
    temp_c=[]
    temp_f=[]
    time_exact=[]
    jy=[]
    jiaoyie={}
    avg_price=0
    down_num=0

    for i in range(data.shape[0]):#对于每一时刻进行讨论
        if i==0:
            continue
        tick_data1=data[i,1:]#取出当前时刻和前一时刻的数据（收盘价、开盘价。。。）
        tick_data2=data[i-1,:]
        month,day,hour,min,second=time_extract(float(data[i,0]))#提取one tick的时间数据
        if month==start_m:#更新当前的时间基准戳,计算平均价格应该从下一时刻起算
            if day!=start_d:
                price_record_day[str(int(start_m))+str(int(start_d))]=temp_ap
                shoupan_rd[str(int(start_m))+str(int(start_d))]=data[i-1,1]
                kaipan_rd[str(int(start_m))+str(int(start_d))]=data[i-1,2]
                down_record_day[str(int(start_m))+str(int(start_d))]=temp_dp
                ceil_rd[str(int(start_m))+str(int(start_d))]=temp_c
                floor_rd[str(int(start_m))+str(int(start_d))]=temp_f
                jiaoyie[str(int(start_m))+str(int(start_d))]=jy
                start_d=day
                temp_ap=[]
                temp_dp=[]
                temp_c=[]
                temp_f=[]
                jy=[]
                continue
        else:
            price_record_day[str(int(start_m))+str(int(start_d))]=temp_ap
            shoupan_rd[str(int(start_m))+str(int(start_d))]=data[i-1,1]
            kaipan_rd[str(int(start_m))+str(int(start_d))]=data[i-1,2]
            down_record_day[str(int(start_m))+str(int(start_d))]=temp_dp
            ceil_rd[str(int(start_m))+str(int(start_d))]=temp_c
            floor_rd[str(int(start_m))+str(int(start_d))]=temp_f
            jiaoyie[str(int(start_m))+str(int(start_d))]=jy
            start_m=month
            start_d=day
            temp_ap=[]
            temp_dp=[]
            temp_c=[]
            temp_f=[]
            jy=[]
            continue
        if tick_data1[-1]!=tick_data2[-1]:
            avg_price=(tick_data1[-2]-tick_data2[-2])/((tick_data1[-1]-tick_data2[-1])*100)#每手100股，计算每股3秒内平均价格
            down_num=tick_data1[-1]-tick_data2[-1]#每3秒内的成交量变化(单位：手)
        price_record.append(avg_price)
        temp_ap.append(avg_price)
        temp_dp.append(down_num)
        temp_c.append(tick_data1[-4])
        temp_f.append(tick_data1[-3])
        jy.append(tick_data1[-2])
        time_exact.append(float(data[i,0]))
        if i==data.shape[0]-1:
            price_record_day[str(int(start_m))+str(int(start_d))]=temp_ap
            down_record_day[str(int(start_m))+str(int(start_d))]=temp_dp
            shoupan_rd[str(int(start_m))+str(int(start_d))]=data[i-1,1]
            kaipan_rd[str(int(start_m))+str(int(start_d))]=data[i-1,2]
            ceil_rd[str(int(start_m))+str(int(start_d))]=temp_c
            floor_rd[str(int(start_m))+str(int(start_d))]=temp_f
            jiaoyie[str(int(start_m))+str(int(start_d))]=jy
    price_record=np.array(price_record)

    #file_save_mat(price_record,'avg_price.mat')
    file_save_csv(price_record_day,shoupan_rd,kaipan_rd,down_record_day,ceil_rd,floor_rd,path)

    return jiaoyie,time_exact

#def MA(t):#移动平均函数，参数t控制

if __name__=="__main__":
    features_append('data/shice_new/601318.SH.xlsx','data/T3_stock_management/601318.SH.xlsx.csv')#如果已经添加过了则应该重新覆写一次
    
    #000333.SZ,300014.SZ,600323.SH,601318.SH