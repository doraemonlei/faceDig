#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/11 9:51
@author: Silence
获取全局纹理特征
'''
import numpy as np
import cv2

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
            print  'kern_size=' + str(ksize) + ', sig=' + str(sigma) + ', th=' + str(theta) + ', lm=' + str(
                lamda) + ', gm=' + str(gam) + ', ps=' + str(ps)
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

def tran_images(path):
    img = cv2.imread(path,1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    img_f = np.array(img, dtype=np.float32)
    img_f /= 255.
    return img_f

def get_gabor_feature(path):
    roi_list = []
    filters = build_filters()
    img = tran_images(path)
    gabor_list = get_gabor(img,filters)
    for x in np.arange(40):
        for i in np.arange(8):
            for j in np.arange(8):
                roi_img = gabor_list[x][(i * 16):((i + 1) * 16), (j * 16):((j + 1) * 16)]
                # print '[%s:%s,%s:%s]' % (i * 16,((i + 1) * 16),j * 16,((j + 1) * 16))
                roi_num = np.sum(roi_img**2)
                roi_list.append(roi_num)

    return roi_list

if __name__ == '__main__':

   get_gabor_feature('../test/2.jpg')