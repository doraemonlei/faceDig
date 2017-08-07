#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/18 16:44
@author: Silence
预处理全部步骤，main函数入口
'''
import cv2
import glob
import time
import os
import dlib
import datetime
# from tools import tool_get_grey_image

def pre_crop_byopencv(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)
    # img = tool_get_grey_image.get_img(path)
    face_cascade = cv2.CascadeClassifier('../tools/haarcascades/haarcascade_frontalface_alt_tree.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    print faces
    crop_face = 0
    for (x, y, w, h) in faces:
        crop_face = img[y:y + h, x:x + w]
    return crop_face

# dlib一幅图片有一张脸
def pre_crop_bydlib(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    crop_face = 0
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        crop_face = img[y1-20:y2+20,x1-20:x2+20]
    return crop_face

# dlib一幅图片有多张脸
def pre_crops_bydlib(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    crop_faces = []
    for face in faces:
        print face
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        crop_face = img[y1 - 20:y2 + 20, x1 - 20:x2 + 20]
        crop_faces.append(crop_face)
    print crop_faces
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
    for path in glob.glob(imagepath):
        c_faces = pre_crops_bydlib(path)
        for c_face in c_faces:
            try:
                z_face = pre_zoom(c_face, 128, 128)
                time.sleep(0.1)
                g_face = pre_gray(z_face)
                clahe_face = pre_create_clahe(g_face)
                filepath, filename = os.path.split(path)
                name, ext = os.path.splitext(filename)
                spath = os.path.join(savepath, name + '_%s.jpg' % time.time())
                print spath
                cv2.imwrite(spath, clahe_face)
            except Exception as e:
                print e


def get_fileName_and_ext(filename):
    filepath,tempfilename = os.path.split(filename)
    shotname,extension = os.path.splitext(tempfilename)
    return shotname

if __name__ == '__main__':
    # t1 = datetime.datetime.now()
    # main(r'../test/ts5.png', r'../test/')
    # t2 = datetime.datetime.now()
    # print (t2-t1).seconds
    # pre_crops_bydlib(r'../test/faces.jpg')
    faces_main(r'../test/ts2.jpg', r'../test/')




