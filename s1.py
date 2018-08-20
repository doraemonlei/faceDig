#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/9/27 16:36
@author: Silence
'''
import datetime
import os
import glob

from preprocessing.pre import process_main
import dataset

max_process = 4

def step_one():
    '''
    人脸识别，图片预处理
    :return:
    '''
    print '*********第一步：人脸识别与图像预处理*********'
    start1 = datetime.datetime.now()
    print '==========================================='
    print '=============开始处理正样本({})=============='.format(len(glob.glob(dataset.POSITIVE_IMAGE)))
    print '图片保存在:{}'.format(dataset.POSITIVE_REC_IMAGE)
    print '==========================================='
    print '                                           '
    process_main(4,dataset.POSITIVE_IMAGE, dataset.POSITIVE_REC_IMAGE)
    end1 = datetime.datetime.now()
    print '用时:{}s'.format((end1 - start1).seconds)

    start2 = datetime.datetime.now()
    print '==========================================='
    print '=============开始处理负样本({})============='.format(len(glob.glob(dataset.NEGATIVE_IMAGE)))
    print '保存在:{}'.format(dataset.NEGATIVE_REC_IMAGE)
    print '==========================================='
    process_main(4,dataset.NEGATIVE_IMAGE, dataset.NEGATIVE_REC_IMAGE)
    end2 = datetime.datetime.now()
    print '用时:{}s'.format((end2 - start2).seconds)

if __name__ == '__main__':
    step_one()