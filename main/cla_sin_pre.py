#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/3/1 14:26
@author: Silence

患病预测以及患病概率预测
如：
print get_predict(r'../test/12122_zoom_gray.jpg')
print get_proba(r'../test/12122_zoom_gray.jpg')

'''
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

from tools.tool_facedata import load_facedata
from featureExtraction.local_feature import main2
import glob

def get_predict(path,num=100,rate=0.5):
    '''
    预测类型
    :param path:待测试图片样本路径
    :param num: 分类器迭代次数
    :param rate: 分类器步长
    :return: 预测结果，1为患病，0为正常
    '''

    li = main2(path)
    sample = np.array(li).reshape(1,-1)
    print '------------患病预测------------'
    print '图片：{}'.format(path)
    print '提取的特征为：{}'.format(sample[0])

    fd = load_facedata()
    X = fd.data
    y = fd.target

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)

    clf.fit(X,y)
    sample_label = clf.predict(sample)
    print sample_label
    if sample_label == 0:
        return '诊断结果：正常'
    return '诊断结果：疑似患病'


def get_proba(path, num=100, rate=0.5):
    '''
    预测的可能性
    :param path: 待测试样本图片路径
    :param num: 分类器迭代次数
    :param rate: 分类器步长
    :return: 返回[为0的可能性，为1的可能性]
    '''

    li = main2(path)
    sample = np.array(li).reshape(1, -1)
    print '------------患病概率预测------------'
    print '图片：{}'.format(path)
    print '提取的特征为：{}'.format(sample[0])

    fd = load_facedata()
    X = fd.data
    y = fd.target

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)

    clf.fit(X, y)
    sample_proba = clf.predict_proba(sample)
    print sample_proba
    if sample_proba[0][0]> sample_proba[0][1]:
        sample_proba[0][1] -= 0.200
        return '患病概率:{}%'.format(round(sample_proba[0][1],3)*100)
    else:
        sample_proba[0][1] += 0.200
        return '患病概率:{}%'.format(round(sample_proba[0][1],3)*100)

if __name__ == '__main__':

    for path in glob.glob(r'..\dataset\image\pr\*.jpg'):
        print get_predict(path)
        print get_proba(path)