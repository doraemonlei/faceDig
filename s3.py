#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/9/28 13:34
@author: Silence
'''
from classification.local_adaboost import get_spec_acc

def step_three():
    '''
    建模预测
    :return:
    '''
    print '***********第三步：建模预测***********'
    get_spec_acc()

if __name__ == '__main__':
    step_three()