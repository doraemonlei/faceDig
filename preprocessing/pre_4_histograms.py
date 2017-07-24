#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
灰度归一化：直方图均衡化
@time: 2016/11/24 15:04
@author: Silence
'''
import cv2
import glob

def pre_create_clahe(img,clipLimit=2,tileGridSize=(10, 10)):
    '''
    CLAHE (Contrast Limited Adaptive Histogram Equalization)对比度自适应直方图均衡化
    :return:
    '''
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    cl1 = clahe.apply(img)

    return cl1

def output_his_image(path):
    '''
    主函数，使用了对比度自适应直方图均衡化的方法，效果比较好
    :param path: 图片路径
    :return: 处理后的图像
    '''
    for imgfile in glob.glob(path):
        print imgfile
        img = cv2.imread(imgfile,0)
        a,b = imgfile.split('.')
        filename = a + '_his.' +b
        img1 = pre_create_clahe(img, clipLimit=1, tileGridSize=(8, 8))
        cv2.imwrite(filename,img1)

if __name__ == '__main__':
    output_his_image('2_gray.jpg')

