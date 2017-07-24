#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/11 16:49
@author: Silence
分类器
'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tools.tool_facedata import load_facedata


def get_predict(sample,num=100,rate=0.5):
    '''
    预测类型
    :param sample:样本5个特征值，[1，2，3，4，5]
    :param num: 分类器迭代次数
    :param rate: 分类器步长
    :return: 预测结果，1为患病，0为正常
    '''
    fd = load_facedata()
    X = fd.data
    y = fd.target

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)

    clf.fit(X,y)
    sample_label = clf.predict(sample)

    return sample_label

def get_proba(sample, num=100, rate=0.5):
    '''
    预测的可能性
    :param sample: 样本5个特征值，[1，2，3，4，5]
    :param num: 分类器迭代次数
    :param rate: 分类器步长
    :return: 返回[为0的可能性，为1的可能性]
    '''
    fd = load_facedata()
    X = fd.data
    y = fd.target

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)

    clf.fit(X, y)
    sample_proba = clf.predict_proba(sample)
    return sample_proba

if __name__ == '__main__':
    print get_predict([228.05,24.02,0.43,3,83.76])
    print get_proba([228.05,24.02,0.43,3,83.76])