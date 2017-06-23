#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
灰度归一化：直方图均衡化
@time: 2016/11/24 15:04
@author: Silence
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob,os

def preCv2AndNumpyShowHistograms():
    '''
    分别使用opencv中的计算直方图方法和numpy中的方法，但是opencv的方法更快
    :return:
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg')  # 直接读为灰度图像
    # opencv方法读取-cv2.calcHist（速度最快）
    # 图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])
    '''
    其中第一个参数必须用方括号括起来。
    第二个参数是用于计算直方图的通道，这里使用灰度图计算直方图，所以就直接使用第一个通道；
    第三个参数是Mask，这里没有使用，所以用None。
    第四个参数是histSize，表示这个直方图分成多少份（即多少个直方柱）。第二个例子将绘出直方图，到时候会清楚一点。
    第五个参数是表示直方图中各个像素的值，[0.0, 256.0]表示直方图能表示像素值从0.0到256的像素。
    最后是两个可选参数，由于直方图作为函数结果返回了，所以第六个hist就没有意义了（待确定）
    最后一个accumulate是一个布尔值，用来表示直方图是否叠加。
    '''
    # numpy方法读取-np.histogram()
    hist_np, bins = np.histogram(img.ravel(), 256, [0, 256])
    # numpy的另一种方法读取-np.bincount()（速度=10倍法2）
    hist_np2 = np.bincount(img.ravel(), minlength=256)
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.plot(hist_cv)
    plt.subplot(223), plt.plot(hist_np)
    plt.subplot(224), plt.plot(hist_np2)
    plt.show()

def preCv2MaskShowHistograms():
    '''
    计算掩膜的直方图
    :return:
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg')  # 直接读为灰度图像
    mask = np.zeros(img.shape[:2], np.uint8)
    print img.shape[:2]
    mask[100:200, 100:200] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # opencv方法读取-cv2.calcHist（速度最快）
    # 图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.show()

def preEqualizeHistograms():
    '''
    全局直方图均衡化
    :return:
    '''
    img = cv2.imread(r'F:\image\imageData\negative\front\c1\gray\22-1_s_zoom_gray.jpg',0)
    res = cv2.equalizeHist(img)

    plt.subplot(121), plt.imshow(img, 'gray')
    plt.subplot(122), plt.imshow(res, 'gray')
    plt.show()

def preCreateClahe(img,clipLimit=2,tileGridSize=(10, 10)):
    '''
    CLAHE (Contrast Limited Adaptive Histogram Equalization)对比度自适应直方图均衡化
    :return:
    '''
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    cl1 = clahe.apply(img)

    return cl1

def outputHisImage(path):
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
        img1 = preCreateClahe(img, clipLimit=1, tileGridSize=(8, 8))
        cv2.imwrite(filename,img1)

if __name__ == '__main__':
    outputHisImage()

