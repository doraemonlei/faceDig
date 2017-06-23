#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2016/11/23 16:45
@author: Silence
'''
import numpy as np
import cv2
def saltAndPepper(src,percetage):
    '''
    为图片添加椒盐噪声
    :param src: 图像，为矩阵
    :param percetage: 椒盐百分比（0-1）
    :return: 处理后的图片
    '''
    noise_img=src
    noise_num=int(percetage*src.shape[0]*src.shape[1])
    for i in range(noise_num):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            noise_img[randX,randY]=0
        else:
            noise_img[randX,randY]=255
    return noise_img

if __name__ == '__main__':
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\77-1_s.jpg')
    img = saltAndPepper(img,0.1)
    cv2.namedWindow('sap',0)
    cv2.imshow('sap', img)
    cv2.waitKey(0)
    cv2.destroyWindow("sap")