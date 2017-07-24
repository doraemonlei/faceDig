#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
灰度处理图像
@time: 2016/11/22 11:05
@author: Silence
'''
import cv2
import glob,os
def pre_gray(path):
    '''
    把图片变成灰度图
    :param path: 图片路径
    :return: 处理后的图像
    '''
    for file in glob.glob(path):
        # print file
        img = cv2.imread(file)
        a,b = file.split('.')
        fileneme = a + '_gray.' + b
        prefile = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(fileneme,prefile)


if __name__ == '__main__':
    # preGray(r'F:\image\imageData4\negative\front\pre\640\*.jpg')
    pre_gray(r'2.jpg')

