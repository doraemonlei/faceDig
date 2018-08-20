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
import numpy as np
import multiprocessing

import dataset

def pre_crop_byopencv(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)
    print img
    # img = tool_get_grey_image.get_img(path)
    face_cascade = cv2.CascadeClassifier('../tools/haarcascades/haarcascade_frontalface_alt_tree.xml')
    '''
    参数1：image--待检测图片，一般为灰度图像加快检测速度；
    参数2：objects--被检测物体的矩形框向量组；
    参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
    参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
        如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
        如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
        这种设定值一般用在用户自定义对检测结果的组合程序上；
    参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
        CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
        因此这些区域通常不会是人脸所在区域；
    参数6、7：minSize和maxSize用来限制得到的目标区域的范围。
    '''
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


def process_main(max_process, imagepath, savepath):
    # max_process = int(sys.argv[0])
    p = multiprocessing.Pool(max_process)
    for i in range(0, max_process):
        p.apply_async(func=faces_main,args=(imagepath,savepath))
    p.close()# 关闭进程池
    p.join()# 等待所有子进程完毕

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
    # faces_main(r'F:\image\imageData5\1000\*.jpg', r'F:\image\imageData5\1000p')
    pre_crop_byopencv(r'C:\Users\Silance\PycharmProjects\faceDig\dataset\image\p\4-TS.JPG')




