#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2016/11/28 10:20
@author: Silence
获得特征点以及标记的图片
'''
import cv2
import dlib
import numpy as np
import glob
# import tool_excel
import openpyxl

PREDICTOR_PATH = "D:\py2.7.12\Lib\site-packages\dlib-19.2.0-py2.7-win-amd64.egg\shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATURE_AMOUNT = 11

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    '''
    本方法主要是获得点的列表
    :param im: 输入一张图像
    :return: 返回68*2的列表
    '''
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annote_landmarks(im, landmarks):
    '''
    在图像上将点表示出来，并且在点的旁边为点标号
    :param im:图像
    :param landmarks:点
    :return:返回图像
    '''
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))

    return im

def read_im_and_landmarks(img):
    '''
    读取图像并获得点
    :param fname: 图像路径
    :return: 图像和点列表
    '''
    # im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(img, (img.shape[1] * SCALE_FACTOR,
                         img.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

if __name__ == '__main__':
    path = r'../test/2.jpg'

    img,landmarks = read_im_and_landmarks(path)
    print path
    print landmarks
    print type(landmarks)

