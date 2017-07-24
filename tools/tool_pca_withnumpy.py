#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2016/11/30 15:13
@author: Silence
使用pca进行特征降维
'''
import numpy as np
import time
import tool_excel_openpyxl

def zeroMean(dataMat):
    meanVal = np.mean(dataMat,axis=0)#按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData,meanVal

def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat, reconMat

def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca2Percentage(dataMat,percentage=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n=percentage2n(eigVals,percentage)                 # 要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)            # 对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   # 最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        # 最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               # 低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  # 重构数据
    return lowDDataMat,reconMat

if __name__ == '__main__':
    # arr = np.arange(100)
    # arr1 = arr.reshape((5,20))
    # print arr1
    # print zeroMean(arr1)

    # rarray = np.random.randint(100,size=(10,10))
    # rarray2 = np.random.random((10,10))
    # print rarray
    # print rarray2
    start = time.clock()
    list = [[1., 1., 11, 1],
            [0.9, 0.95, 1.1, 1],
            [1.01, 1.03, 2.6, 1],
            [2., 2., 55, 1],
            [2.03, 2.06, 6.5, 1],
            [1.98, 1.89, 3, 1],
            [3., 3., 0, 1],
            [3.03, 3.05, 111, 1],
            [2.89, 3.1, 1.9, 1],
            [4., 4., 2.7, 1],
            [4.06, 4.02, 8.7, 1],
            [3.97, 4.01, 9.5, 1.1]]
    # data = np.array(list)
    data = tool_excel_openpyxl.readExcel2Nparray(r'F:\image\imageData\sum_normalization_distance.xlsx')
    a,b = pca2Percentage(data)
    c = np.array(a,dtype=np.float64)
    print c
    # for i in c:
    #     print i


    end = time.clock()
    print "read: %f s" % (end - start)



