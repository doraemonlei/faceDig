#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/14 13:31
@author: Silence
人脸检测通过opencv
'''
from tool_get_grey_image import get_img

import cv2

def face_dect(image):
    face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    return faces

def eye_dect(image):
    eye_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_eye_tree_eyeglasses.xml')
    eyes = eye_cascade.detectMultiScale(image, 1.1, 5)
    return eyes

if __name__ == '__main__':
    img = get_img('../test/1.jpg')
    print img
    faces = face_dect(img)
    print faces