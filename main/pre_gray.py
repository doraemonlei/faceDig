#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/3/1 11:58
@author: Silence

灰度归一化
灰度变换后进行直方图均衡化处理

如：
pre_gray(r'../test/12122_zoom_gray.jpg')
pre_gray(r'../test/*.jpg')
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

        # clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
        # clahe_face = clahe.apply(prefile)

        # cv2.imwrite(fileneme,clahe_face)
        cv2.imwrite(fileneme,prefile)

if __name__ == '__main__':
    pre_gray(r'C:\Users\Silance\Desktop\12122_zoom.jpg')
    # print cv2.__version__