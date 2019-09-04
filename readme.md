代码库中代码根据以下几篇论文实现，主要是针对鲁棒性机器学习算法及其分布式实现。可以作为鲁棒性机器学习算法的应用。引用请参考如下论文：  

Lei, Dajiang, Hongyu Zhang, Hongtao Liu, Zhixing Li and Yu Wu. "Maximal Uncorrelated Multinomial Logistic Regression." IEEE Access 7,  (2019): 89924-89935.  

Lei, Dajiang, Jianyang Tang, Zhixing Li and Yu Wu. "Using Low-Rank Approximations to Speed up Kernel Logistic Regression Algorithm." IEEE Access 7,  (2019): 84242-84252.
Lei, Dajiang, Zhijie Jiang and Yu Wu. "Weighted Huber Constrained Sparse Face Recognition."  (2019).  

Lei, Dajiang, Zhijie Jiang, Meng Du, Hao Chen and Yu Wu. "Large Scale Huber Regression." In 8th Annual IEEE International Conference on Cyber Technology in Automation, Control and Intelligent Systems, CYBER 2018, July 19, 2018 - July 23, 2018, 1282-1287. Tianjin, China: Institute of Electrical and Electronics Engineers Inc., 2019.  

Lei, Dajiang, Meng Du, Hao Chen, Zhixing Li and Yu Wu. "Distributed Parallel Sparse Multinomial Logistic Regression." IEEE Access 7,  (2019): 55496-55508.

## ADMM求解稀疏多元逻辑回归模型
### <strong>对于一致性问题，</strong> 程序入口为:  
src/SMLR/src/test/scala/classification/ConsensusSMLRTest.scala  

数据集：COIL20 采用五折交叉验证

主要参数说明与设定如下：    
<strong>maxItertimes</strong>: 50, ADMM框架迭代次数  
<strong>conThreshold</strong>: 0.01, 原始残差和对偶残差的阈值，同时小于时即完成训练  
<strong>lambda</strong>: 0.0001, 正则系数  
<strong>rho</strong>: 0.001, 增广拉格朗日引入的系数  
<strong>numPartition</strong>: 6, 并行块数

### <strong>对于共享问题,</strong> 程序入口为:  
src/SMLR/src/test/scala/classification/SharingSMLRTest.scala

数据集：COIL20 采用五折交叉验证  

主要参数说明与设定如下：  
<strong>maxItertimes</strong>: 50, ADMM框架迭代次数  
<strong>conThreshold</strong>: 0.01, 原始残差和对偶残差的阈值，同时小于时即完成训练  
<strong>lambda</strong>: 0.01, 正则系数  
<strong>rho</strong>: 0.01, 增广拉格朗日引入的系数  
<strong>numPartition</strong>: 6, 并行块数


## ADAM串行求解极大无关多元逻辑回归模型
程序入口为：  
src/UMLR/adam/UMLR.py   

数据集：image.csv

主要参数说明与设定如下：  
<strong>epochs</strong>: 1000, adam迭代次数  
<strong>learning_rate</strong>: 0.001, 学习率.   
<strong>reg_strength</strong>: 0.00001, reg_strength   
<strong>eta</strong>: 10*1e-5, eta

## ADMM并行求解极大无关多元逻辑回归
### <strong>对于一致性问题，</strong> 程序入口为:  
src/UMLR/admm/src/main/scala/org/harvey/ml/classification/MLR_Consensus.scala

数据集：iris.csv

主要参数说明与设定如下：  
<strong>epochs</strong>: 400, adam迭代次数.  
<strong>rho</strong>: 0.01, 增广拉格朗日引入的系数  
<strong>eta</strong>: 0.00001, eta  
<strong>lambda</strong>: 0.00001, 正则系数  

### <strong>对于共享问题，</strong> 程序入口为:  
src/UMLR/admm/src/main/scala/org/harvey/ml/classification/MLR_Sharing.scala  


数据集：iris.csv

主要参数说明与设定如下：  
<strong>epochs</strong>: 300, adam迭代次数.  
<strong>rho</strong>: 0.001, 增广拉格朗日引入的系数  
<strong>eta</strong>: 0.00001, eta  
<strong>lambda</strong>: 0.00001, 正则系数  

## ADMM串行求解WHCSC人脸识别模型
程序入口：  
src/WHCSC/WHCSC.py 

数据集：YLB_ 32_38  


