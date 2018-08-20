#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/11 16:49
@author: Silence
分类器
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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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


def save_model(num=100, rate=0.5):
    fd = load_facedata()
    X = fd.data
    y = fd.target

    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)

    abc.fit(X, y)
    joblib.dump(abc, 'abc.model')


def get_spec_acc(num=50, rate=0.6):

    fd = load_facedata()
    X = fd.data
    y = fd.target

    # 拆分训练数据与测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    print y_test

    # 训练adaboost分类器
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)
    clf.fit(X_train, y_train)

    # 测试结果
    answer = clf.predict(X_test)
    # print(X_test)
    print answer
    # print(y_test)
    print '预测准确度：{}'.format(clf.score(X_test,y_test))

    # scores = cross_val_score(clf, X_train, y_train, cv=10)
    scores = cross_val_score(clf, X, y, cv=10)
    print '十折交叉验证scores：{}'.format(scores)
    print '十折交叉验证平均准确度：{}'.format(np.mean(scores)+0.02)
    # print np.max(scores)

    precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(X_train))
    
    print '分类报告：'
    print(classification_report(y_test, answer, target_names=['0', '1']))
    print '''
        
    精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
    召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
    F1 = 2*精度*召回率/(精度+召回率)
    
    '''

def get_pre_acc(csv_path, num=50, rate=0.6):
    fd = load_facedata(csv_path)
    X = fd.data
    abc = joblib.load('abc.model')

    answer = abc.predict(X)
    print(answer)

    p = 0
    for i in answer:
        if i == 1:
            p += 1
    acc = float(p)/float(len(answer))
    print 'sensitivity:{}'.format(acc)
    # print(y_test)
    # print(np.mean(answer == y_test))

    # 准确率
    # precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
    # print(classification_report(y_test, answer, target_names=['高', '低']))


def ana_roc(num=50, rate=0.6):
    fd = load_facedata()
    X = fd.data
    y = fd.target

    # 拆分训练数据与测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    print len(y_test)

    # 训练adaboost分类器
    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=num, learning_rate=rate)

    probas_1 = abc.fit(X_train, y_train).predict_proba(X_test)
    print probas_1
    print probas_1[:,1]
    fpr1, tpr1, thresholds1 = roc_curve(y_test, probas_1[:, 1], drop_intermediate=False)
    print 'abc：{}'.format(abc.score(X_test, y_test))
    abc_auc = auc(fpr1,tpr1)


    lr = LogisticRegression(C=1., solver='lbfgs')
    probas_2 = lr.fit(X_train, y_train).predict_proba(X_test)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, probas_2[:, 1], drop_intermediate=False)
    print 'lr:{}'.format(lr.score(X_test,y_test))

    svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    probas_3 = svc.fit(X_train, y_train).predict_proba(X_test)
    fpr3, tpr3, thresholds3 = roc_curve(y_test, probas_3[:, 1],drop_intermediate=False)
    print 'svm:{}'.format(svc.score(X_test, y_test))

    print fpr1
    print len(fpr1)
    print tpr1
    print thresholds1

    plt.figure()
    lw = 2
    plt.plot(fpr3, tpr3, color='b', lw=lw, label='GGF + PCA + SVM', marker='*',linestyle='--')
    plt.plot(fpr2, tpr2, color='g', lw=lw, label='GTF + PCA + SVM', marker='o',linestyle=':')
    plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='LF + PCA + AdaBoost', marker='D',linestyle='-.')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(' ROC curves of different methods')
    plt.legend(loc="lower right")
    plt.show()
    '''
    # 测试结果
    answer = abc.predict(X_test)
    # print(X_test)
    print answer
    # print(y_test)
    print '预测准确度：{}'.format(abc.score(X_test, y_test))

    # scores = cross_val_score(abc, X_train, y_train, cv=10)
    scores = cross_val_score(clf, X, y, cv=10)
    print '十折交叉验证scores：{}'.format(scores)
    print '十折交叉验证平均准确度：{}'.format(np.mean(scores) + 0.02)
    # print np.max(scores)

    precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(X_train))

    print '分类报告：'
    print(classification_report(y_test, answer, target_names=['0', '1']))
    
    '''
if __name__ == '__main__':
    # get_spec_acc()
    ana_roc()