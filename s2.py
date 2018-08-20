#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/9/30 13:09
@author: Silence
'''

import datetime
import os

from preprocessing.pre import faces_main
from tools.tool_csv import input_csv,ainput_csv
import dataset

def step_two():
    '''
    局部特征提取
    :return:
    '''
    print '***********第二步：LF特征提取***********'
    print '=============开始提取正样本============'
    print '数据存储位置：dataset\dataface.csv'
    input_csv(os.path.join(dataset.POSITIVE_REC_IMAGE,'*.jpg'),'dataset\dataface.csv',242)
    print '==============特征提取完成============='
    print '                                     '
    print '=============开始提取负样本============'
    ainput_csv(os.path.join(dataset.NEGATIVE_REC_IMAGE,'*.jpg'),'dataset\dataface.csv')
    print '==============特征提取完成============='

if __name__ == '__main__':
    step_two()