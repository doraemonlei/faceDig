#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/13 15:55
@author: Silence
获得灰度图像
'''
import cv2

def get_img(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

if __name__ == '__main__':
    get_img()