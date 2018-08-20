#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/3/1 11:52
@author: Silence

图片几何归一化
将图片同意变为640*640尺寸

如：
pre_zoom(r'../test/12122_zoom_gray.jpg',640,640)
pre_zoom(r'../test/*.jpg',640,640)
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
        fileneme = a+'_ge.'+b
        prefile = cv2.resize(img,(height,width))
        cv2.imwrite(fileneme,prefile)

if __name__ == '__main__':
    pre_zoom(r'C:\Users\Silance\Desktop\12122.jpg',640,640)