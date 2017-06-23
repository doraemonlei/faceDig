#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
变换图像尺寸
@time: 2016/11/23 14:38
@author: Silence
'''
import cv2
import glob,os
def preZoom(path,width,height):
    '''
    图像尺寸变换
    :param path:文件，图像的路径
    :param width: 变换后的宽
    :param height: 变换后的高
    :return: 返回变换后的图像，保存在当前目录下
    '''
    for file in glob.glob(path):
        # print file
        img = cv2.imread(file)
        a,b = file.split('.')
        fileneme = a+'_zoom.'+b
        prefile = cv2.resize(img,(width,height))
        cv2.imwrite(fileneme,prefile)


if __name__ == '__main__':
    preZoom()