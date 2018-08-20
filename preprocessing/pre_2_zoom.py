#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
变换图像尺寸
@time: 2016/11/23 14:38
@author: Silence
'''
import cv2
import glob,os
def pre_zoom(path,width,height):
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
    img = cv2.imread(r'C:\Users\Silance\PycharmProjects\faceDig\test\1.jpg')
    h = img.shape[0]/2
    w = img.shape[1]/2
    print img.shape
    print h,w
    pre_zoom(r'C:\Users\Silance\PycharmProjects\faceDig\test\1.jpg',w,h)