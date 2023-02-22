from pandas import *#该文件收集特征构造函数集
import os
import numpy as np

class totalfunc():#包装所有需要用到的函数:
                #价格MA(6,12,24),成交量MA(3,6,12),价格涨速,成交量涨速，KDJ,MACD,BOLL
    initia=True
    def __init__(self):
        print('func find')
            
    def jiage_MA(self,p,win):#接收参数是ndarray,size一般是(dim,1),即dim维的序列，最后一个元素是你当前的时刻。函数最终返回一个数，即特征指标
        sum=0.0
        p=p.tolist()#先把ndarrays转换成list,便于后续操作
        p=p[::-1]#逆序list
        if (len(p)<24 and win==24) or (len(p)<12 and win==12) or (len(p)<6 and win==6):
            for i in range(len(p)):
                sum+=p[i]
            return sum/(len(p))
        else:
            for i in range(win):
                sum+=p[i]
            sum=sum/win
            return sum
    
    def chengjiaoliang_MA(self,cjl,win):
        sum=0.0
        cjl=cjl.tolist()#先把ndarrays转换成list,便于后续操作
        cjl=cjl[::-1]#逆序list
        if (len(cjl)<24 and win==24) or (len(cjl)<12 and win==12) or (len(cjl)<6 and win==6):
            for i in range(len(cjl)):
                sum+=cjl[i]
            return sum/(len(cjl))
        else:
            for i in range(win):
                sum+=cjl[i]
            sum=sum/win
            return sum
    
    def p_climb_rate(self,p):#当前时刻价格和上一tick价格的比值
        return p[-1]/p[-2]

    def c_climb_rate(self,cjl):#当前时刻成交量和上一tick成交量的比值
        return cjl[-1]/cjl[-2]
    
    def KDJ(self,seq,shoupan,paras=[50,50],s=9):#接收参数为一段序列（平均价格）、当日收盘价、前一时刻的K、D值（列表）、周期(默认值为9)。若无前一日K、D值，都默认为50
        ceil=seq[np.argmax(seq)]#求前一个周期内(包括当前时刻)的最低价和最高价
        floor=seq[np.argmin(seq)]
        if ceil!=floor:
            RSV=((shoupan-floor)/(ceil-floor))*100
        else:
            RSV=0
        K=(2/3)*paras[0]+(1/3)*RSV
        D=(2/3)*paras[1]+(1/3)*K
        J=3*K-2*D
        return K,D,J
    def normalization(self,data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range
 
 
    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        if sigma.any()==0:
            print('error')
        return (data - mu)/sigma

    def earning_rate(self,mod,P1,P2,deal,s=0.00015):#计算收益率（p1为开仓价格，p2为平仓价格,deal为成交量,s为手续费比例)，开头需要传入交易类型（做多1或做空2）
        y=0.001#印花税
        deal=deal*100#传入的交易量单位为手，因此乘以100转换为股
        if mod==1:#做多
            return (P2*deal-P1*deal-P1*deal*s-P2*deal*s-P2*y)/(P1*deal)
        elif mod==2:#做空
            return (P1*deal-P2*deal-P1*deal*s-P2*deal*s-P1*y)/(P1*deal)

    def earning(self,mod,P1,P2,deal,s=0.00015):#计算收益
        y=0.001#印花税
        deal=deal*100#传入的交易量单位为手，因此乘以100转换为股
        if mod==1:#做多
            return (P2*deal-P1*deal-P1*deal*s-P2*deal*s-P2*y)
        elif mod==2:#做空
            return (P1*deal-P2*deal-P1*deal*s-P2*deal*s-P1*y)

    def cov(self,x1,x2,price1,price2,kaipan):#计算协方差。接收两个列表(包含利润率序列),以及两个股价序列,和当天开盘价
        x1=np.array(x1)
        x2=np.array(x2)
        price1=np.array(price1)
        price2=np.array(price2)
        n1=x1.shape[0]
        n2=x2.shape[0]
        jump1=np.zeros((n1))
        jump2=np.zeros((n2))
        k=0.04
        for i in range(1,n1):
            if (price1[i]/kaipan)>=k:
                jump1[i]=1
            else:
                jump1[i]=0
        for i in range(1,n2):
            if (price2[i]/kaipan)>=k:
                jump2[i]=1
            else:
                jump2[i]=0
        cov=0
        for i in range(n1-1):
            for j in range(n2-1):
                cov+=(x1[i+1]-x1[i])*(x2[j+1]-x2[j])*jump1[i+1]*jump2[j+1]
        return cov

    def v(self,x):#计算方差
        x=np.array(x)
        return np.var(x)
    def E(self,x):#计算某向量的期望
        x=np.array(x)
        return np.mean(x)

    def weight_reconstruct(self,w):
        s=np.sum(w)
        for i in range(w.shape[0]):
            w[i]=w[i]/s
        return w
    #def MACD():

    #def BOLL(seq,M):##接收参数为一段序列
        