#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
人脸校正，剪裁
@time: 2017/6/23 15:33
@author: Silence
'''
# !/usr/bin/env python

import cv2

def imgCrop(path,imgName):
    '''
    人脸剪裁
    :param path:图片路径
    :param imgName: 剪裁后保存路径及文件名
    :return: 保存图像
    '''
    face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        x1 = x-20
        y1 = y-20
        x2 = x+w+40
        y2 = y+h+40
        crop = img[y1:y2,x1:x2]

    cv2.imwrite(imgName, crop)

if __name__ == '__main__':
    imgCrop()