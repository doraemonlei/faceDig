#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/3/1 10:18
@author: Silence

预处理全部流程，识别出人脸后根据识别的矩形框将人脸剪裁下来。

faces_main(imagepath,savepath)主函数
参数1 图片路径，可以是单一图片，也可以为匹配格式
参数2 保存文件夹路径
return 保存处理后的图片

如：
faces_main(r'../test/12122_zoom_gray.jpg', r'../test')
faces_main(r'../test/*.jpg', r'../test')
'''
import cv2
import glob
import time
import os
import dlib
import datetime
import numpy as np
import multiprocessing

import dataset

# dlib检测，一幅图片有多张脸，分别截取
def pre_crops_bydlib(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    crop_faces = []
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        crop_face = img[y1 - 20:y2 + 20, x1 - 20:x2 + 20]
        crop_faces.append(crop_face)
    return crop_faces

def pre_zoom(img,width,height):

    zoom_face = cv2.resize(img,(height,width))
    return zoom_face

def pre_gray(img):

    gray_face = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray_face

def pre_create_clahe(img):

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    clahe_face = clahe.apply(img)

    return clahe_face

def faces_main(imagepath,savepath):
    count = 1
    for path in glob.glob(imagepath):
        num = len(glob.glob(imagepath))
        print '----------({}/{})开始处理图片{}----------'.format(count,num,path)
        print '人脸识别...'
        start = datetime.datetime.now()
        c_faces = pre_crops_bydlib(path)
        end = datetime.datetime.now()
        print '共检测出{}张脸...'.format(len(c_faces))
        print '人脸识别耗时{}秒'.format((end - start).seconds)
        count += 1
        for c_face in c_faces:
            try:
                print '几何归一化...'
                z_face = pre_zoom(c_face, 640, 640)
                time.sleep(0.1)
                print '灰度归一化...'
                g_face = pre_gray(z_face)
                print '直方图均衡化...'
                clahe_face = pre_create_clahe(g_face)
                filepath, filename = os.path.split(path)
                name, ext = os.path.splitext(filename)
                spath = os.path.join(savepath, name + '_%s.jpg' % time.time())
                cv2.imwrite(spath, clahe_face)
                print '                                           '
            except Exception as e:
                print e

if __name__ == '__main__':

    faces_main(r'..\dataset\image\n\*.JPG', r'..\dataset\image\nr')


