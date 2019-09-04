# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from numpy import *;
import numpy as np;
import os

def CalculateMAE(predictValue,testY):
    return np.sum([np.fabs(predictValue[i]-testY[i]) for i in range(0,len(testY)-1)])/len(testY)

def CalculateMAE_1(predictValue):
    return np.sum([np.fabs(predictValue[i]) for i in range(0,len(predictValue)-1)])/len(predictValue)

def Shrinkage1(k,a):
    k=np.array(k.T.tolist()[0])
    a=np.array(a.T.tolist()[0])

    a_jia_k=a+k
    a_jian_k=a-k

    a_jia_k[np.where(a_jia_k >= 0)]=0
    a_jian_k[np.where(a_jian_k <= 0)]=0

    return np.mat(a_jia_k+a_jian_k).T

def Shrinkage2(K,a):
    a=np.array(a.T.tolist()[0])

    a_ = a.copy()
    a_[np.where((a_ <= K) & (a_ >= -K))] = 0
    a_[np.where(a_ > K)] -= K
    a_[np.where(a_ < - K)] += K

    return np.mat(a_).T

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = (data[size//2]+data[size//2-1])/2
        data[0] = median
    if size % 2 == 1:
        median = data[(size-1)//2]
        data[0] = median
    return data[0]

def ADMM(trainX,trainY,Wei):
    #theta_temp=ADMM_1(trainX,trainY)
    A=trainX
    b=trainY

    RHO1,RHO2,Lamda=1,1,1
    ETA=10*Wei
    # ETA=np.array(ETA)[0][0]

    #ETA=15
    Z=np.mat(np.ones(trainX.shape[0])).T
    W=np.mat(np.ones(trainX.shape[1])).T
    Uz=np.mat(np.ones(trainX.shape[0])).T
    Uw=np.mat(np.ones(trainX.shape[1])).T
    theta=0
    diff=1
    count=1
    while count<=2:
        X1=np.linalg.pinv(RHO1*A.T*A+RHO2)*(RHO1*A.T*(Z-Uz+b)+RHO2*(W-Uw))
        #print X1.tolist()
        Z1=(RHO1/(1+RHO1))*(A*X1-b+Uz)+Shrinkage1((ETA/RHO1),(1/(1+RHO1))*(A*X1-b+Uz))
        W1=Shrinkage2(Lamda/RHO2,X1+Uw)
        Uz1=Uz+A*X1-Z1-b
        Uw1=Uw+X1-W1
        Z=Z1
        W=W1
        Uz=Uz1
        Uw=Uw1
        # diff=np.sum(np.square(theta-X1))
        # print diff,X1.tolist()
        theta=X1
        count+=1
    return theta

def Iteration(trainX,trainY,t,c,n):
    trainX=np.mat(trainX.drop(n,1).values.tolist()).T
    trainY=np.mat(trainY).T

    ##权重初始化
    residual=np.square(trainY-np.mean(trainX,axis=1))
    residual_sort=np.sort(residual,axis=0)
    iter=residual_sort[int(len(residual_sort)*t)].tolist()[0][0]
    beta=c/iter
    temp=-beta*(residual-iter)
    # for k in range(len(temp)):
    #     if temp[k]<-700:
    #         temp[k]=-700
    EXP=exp(temp)
    w=EXP/(1+EXP)
    #Step_length=0.01
    iteration=1
    diff=0.1
    theta=0
    while iteration<=2:
        W=np.sqrt(np.diag(w.T.tolist()[0]))

        theta=ADMM(W*trainX,W*trainY,np.sqrt(w))

        residual=np.square((trainY-trainX*theta))
        residual_sort=np.sort(residual,axis=0)
        iter=residual_sort[int(len(residual_sort)*t)].tolist()[0][0]
        beta=c/iter
        temp=-beta*(residual-iter)
        # for k in range(len(temp)):
        #     if temp[k]<-700:
        #         temp[k]=-700
        EXP=exp(temp)
        w=EXP/(1+EXP)

        iteration+=1
    # print (W*(trainY-trainX*theta)).T.tolist()[0]
    return (W*np.abs(trainY-trainX*theta)).T.tolist()[0]


def main():
    people_count=38
    n=896
    t=0.7
    c=1

    ##训练集预处理
    #traindata=pd.read_table('D:\\FaceRecognition\\finall_YLB_32_28\\train_YB_32_28_noise_00.csv',header=None,sep=',')
    traindata=pd.read_table('/Users/ch_cmpter/Desktop/2016级资料存档/蒋志杰/蒋志杰资料（论文代码、参考文献、人脸数据集）/FaceRecognition/finall_YLB_32_28/train_YB_32_28_noise_00.csv',header=None,sep=',')

    ##测试集预处理
    #testdata=pd.read_table('D:\\FaceRecognition\\finall_YLB_32_28\\test_YB_32_28_noise_50.csv',header=None,sep=',')
    testdata=pd.read_table('/Users/ch_cmpter/Desktop/2016级资料存档/蒋志杰/蒋志杰资料（论文代码、参考文献、人脸数据集）/FaceRecognition/finall_YLB_32_28/test_YB_32_28_noise_50.csv',header=None,sep=',')
    testdata_label=testdata.get(n)
    testdata=testdata.drop(n,1)
    testdata=testdata.values.tolist()



    #for i in range(len(testdata)):
    label_list=[]
    result=[]
    for i in range(len(testdata)):
        Mae=[]
        for j in range(people_count):
            #print i,j
            train_x=traindata[traindata[n]==j+1]
            test_y=testdata[i]
            predictValue=Iteration(train_x,test_y,t,c,n)
            result.append(predictValue)
            Mae.append(CalculateMAE_1(predictValue))
        print(Mae.index(np.min(Mae))+1,testdata_label[i])
        label_list.append(Mae.index(np.min(Mae))+1)
        result=pd.DataFrame(result)
        result.to_csv('4_result_2.csv',index=False,sep=',')
    true=0
    for i in range(len(label_list)):
        if testdata_label[i]==label_list[i]:
            true+=1
    print(true/len(label_list))
    print(os.path.basename(__file__))



if (__name__=="__main__"):
    main()
