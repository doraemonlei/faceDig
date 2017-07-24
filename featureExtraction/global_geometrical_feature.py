#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/11 9:51
@author: Silence
获取全局几何特征
'''
from tools import tool_get_landmarksAndImages
from tools import tool_get_grey_image
import numpy as np

def getdis(nparray):
    '''
    获得欧氏距离
    :param nparray:输入一个nparray
    :return: 返回距离列表
    '''
    a = []
    for i in range(len(nparray)):
        for j in range(len(nparray)):
            if i != j and i < j:
                a.append('%.2f' % np.linalg.norm(nparray[i] - nparray[j]))
    a = map(float,a)
    temp = []
    for i in a:
        temp.append(i)
    return temp

def maxMinNormalization(x, Max, Min):
    '''
    0,1归一化
    :param x:输入的数
    :param Max: 最大值
    :param Min: 最小值
    :return: 归一化的值
    '''
    x = (x - Min) / (Max - Min)
    return x

def normalizationToArray(nparray):
    '''
    将归一化的数添加到一个列表中
    :param nparray:
    :return:
    '''
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

def main(path):
    img = tool_get_grey_image.get_img(path)
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)
    distance = getdis(landmarks)
    normalizationDditance = normalizationToArray(distance)
    return normalizationDditance
if __name__ == '__main__':
    print main(path = r'../test/2.jpg')
