#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/18 14:27
@author: Silence
获取gabor纹理特征，使用opencv
'''
import numpy as np
import cv2
from tool_crop_eyesArea import get_eyes_area
from tool_normalization import normalizationToArray
from tool_get_grey_image import get_img

# 构建Gabor滤波器
def build_filters():
    ksize = 21
    sigma = 2.0 * np.pi
    gam = 1.0
    ps = -90 * np.pi / 180
    filters = []
    # ksize = [9,11,13,15,17] # gabor尺度，5个
    # theta = [i for i in np.arange(0, np.pi, np.pi/8)] #gabor方向 8个
    lamda = [17,18,19,20,21] #波长

    for lamda in lamda:
        for theta in np.arange(0, np.pi, np.pi/8):
            kern = cv2.getGaborKernel((ksize, ksize),sigma, theta, lamda, gam, ps)
            filters.append(kern)
            # print  'kern_size=' + str(ksize) + ', sig=' + str(sigma) + ', th=' + str(theta) + ', lm=' + str(
            #     lamda) + ', gm=' + str(gam) + ', ps=' + str(ps)
    return filters

def get_gabor(img,filters):
    res = []  # 滤波结果
    for i in xrange(len(filters)):
        res1 = cv2.filter2D(img, cv2.CV_32F, filters[i])
        res.append(np.asarray(res1))
    return res

def show_gabor_image(path):
    filters = build_filters()
    imgs = tran_images(path)
    res = get_gabor(imgs,filters)
    cv2.imshow('fuzhi', res[6][0:15, 0:15])
    cv2.imshow('mag', np.power(res[6], 2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tran_images(img):
    eye_area = get_eyes_area(img)
    img_f = np.array(eye_area, dtype=np.float32)
    img_f /= 255.
    return img_f

def get_gabor_feature(img):
    roi_list = []
    filters = build_filters()
    img = tran_images(img)
    gabor_list = get_gabor(img,filters)
    for x in np.arange(40):
        for i in np.arange(8):
            roi_img = gabor_list[x][0:50, (i * 40):((i + 1) * 40)]
            # print '[%s:%s,%s:%s]' % (i * 16,((i + 1) * 16),j * 16,((j + 1) * 16))
            roi_num = np.sum(roi_img**2)
            roi_list.append(roi_num)
    nor_list = normalizationToArray(roi_list)
    eye_gabor = np.sum(nor_list)
    return eye_gabor

if __name__ == '__main__':
    img = get_img('../test/ts2.jpg')
    print get_gabor_feature(img)
    # print np.max(get_gabor_feature('../test/ts2.jpg'))
    # print np.min(get_gabor_feature('../test/ts2.jpg'))