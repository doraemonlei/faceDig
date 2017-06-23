#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
图像滤波，由于效果不好，没有使用
@time: 2016/11/23 16:36
@author: Silence
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pre_saltandpepper
'''
对于2D图像可以进行低通或者高通滤波操作:
低通滤波（LPF）有利于去噪，模糊图像，
高通滤波（HPF）有利于找到图像边界。

'''

def pre2DFilter():
    '''
    自定义核滤波
    :return:显示图像
    '''
    img = cv2.imread(r'69-1.jpg')
    # kernel1 = np.ones((5,5),np.float32)/25
    kernel2 = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), np.float32)
    kernel3 = np.array(([-1, -1, -1], [-1, 9, -1], [-1, -1, -1]), np.float32)
    kernel4 = np.array(([1, -2, 1], [-2, 5, -2], [1, -2, 1]), np.float32)
    # dst = cv2.filter2D(img,-1,kernel1)
    dst1 = cv2.filter2D(img, -1, kernel2)
    dst2 = cv2.filter2D(img, -1, kernel3)
    dst3 = cv2.filter2D(img, -1, kernel4)

    plt.subplot(221), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(dst1), plt.title('kernel1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(dst2), plt.title('kernel2')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(dst3), plt.title('kernel3')
    plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.imwrite('kernel1.jpg', dst1)
    cv2.imwrite('kernel2.jpg', dst2)
    cv2.imwrite('kernel3.jpg', dst3)


def preAveFilter():
    '''
    cv2.blur和cv2.boxfilter
    :return: 显示图像
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg')
    blur = cv2.blur(img,(5,5))
    boxfilter = cv2.boxFilter(img,ddepth=-1,ksize=(3,3),normalize=False)

    plt.subplot(131), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(blur), plt.title('Blur')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(boxfilter), plt.title('BoxFilter')
    plt.xticks([]), plt.yticks([])
    plt.show()

def preGaussianBlur():
    '''
    高斯滤波
    :return: 显示图像
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg')
    img = pre_saltandpepper.saltAndPepper(img,0.01)
    blur = cv2.GaussianBlur(img,(5,5),0)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('GaussianBlur')
    plt.xticks([]), plt.yticks([])
    plt.show()

def preMedianBLur():
    '''
    中值滤波
    :return: 显示图像
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg')
    img = pre_saltandpepper.saltAndPepper(img, 0.1)
    blur = cv2.medianBlur(img,5)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('MedianBLur')
    plt.xticks([]), plt.yticks([])
    plt.show()

def preBilateralFilter():
    '''
    双边滤波
    cv2.bilateralFilter(img,d,’p1’,’p2’)函数有四个参数，
    d：是领域的直径
    p1：空间高斯函数标准差
    p2：灰度值相似性高斯函数标准差。
    :return:
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg')
    img = pre_saltandpepper.saltAndPepper(img, 0.01)
    blur = cv2.bilateralFilter(img,9,75,75)
    #9:滤波领域直径,75:空间高斯函数标准差,75:灰度值相似性标准差
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('BilateralFilter')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    pre2DFilter()
    # preAveFilter()
    # preGaussianBlur()
    # preMedianBLur()
    # preBilateralFilter()