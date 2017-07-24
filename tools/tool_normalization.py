#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2016/11/30 11:03
@author: Silence
归一化数据
'''
import numpy as np

def maxMinNormalization(x, Max, Min):

    x = (x - Min) / (Max - Min)
    return x

def normalizationToArray(nparray):
    array = np.array(nparray)
    Max = np.max(array)
    Min = np.min(array)

    list = []
    for i in array:
        list.append('%.2f' % maxMinNormalization(i, Max, Min))

    list = map(float, list)
    temp = []
    for i in list:
        temp.append(i)

    return temp