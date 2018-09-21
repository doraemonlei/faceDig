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
import datetime
import dlib
import numpy as np
import multiprocessing
from multiprocessing import Pool

import dataset

PREDICTER_5_PATH = dataset.LANDMARKS_5_PATH
PREDICTER_68_PATH = dataset.LANDMARKS_68_PATH

def face_crop_byopencv(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    print(img)
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
    print(faces)
    crop_face = 0
    for (x, y, w, h) in faces:
        crop_face = img[y:y + h, x:x + w]
    return crop_face


# 人脸校正path
def faces_pre_path_alignment(path):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTER_5_PATH)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(path))
        exit()

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(img, faces, size=640, padding=0.25)

    return images

# 人脸校正path
def faces_pre_imgs_alignment(imgs):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTER_5_PATH)

    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(imgs, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(path))
        exit()

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(imgs, detection))

    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(imgs, faces, size=640, padding=0.25)

    return images

# dlib一幅图片有多张脸，参数为图像路径
def faces_pre_crops_path_bydlib(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)
    crop_faces = []
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        crop_face = img[y1 - 100:y2 + 100, x1 - 100:x2 + 100]
        crop_faces.append(crop_face)
    return crop_faces


# dlib一幅图片有多张脸，参数为已读取的图像
def faces_pre_crops_imgs_bydlib(imgs):
    crop_faces = []
    for img in imgs:
        detector = dlib.get_frontal_face_detector()
        faces = detector(img, 1)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # crop_face = img[y1 - 20:y2 + 20, x1 - 20:x2 + 20]
            crop_face = img[y1:y2, x1:x2]
            crop_faces.append(crop_face)
    return crop_faces


# 几何归一化
def face_pre_zoom(img, width, height):
    zoom_face = cv2.resize(img, (height, width))
    return zoom_face


# 灰度归一化
def face_pre_gray(img):
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_face


# 直方图均衡化
def face_pre_histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    clahe_face = clahe.apply(img)
    return clahe_face


# 获取所有待处理图像路径
def face_pre_get_image_path(imagedir):
    return glob.glob(os.path.join(imagedir, '*'))


# 人脸图像预处理总步骤
def faces_main(imagepath, imagedir, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    imgs = face_pre_get_image_path(imagedir)
    num = len(imgs)

    count = imgs.index(imagepath) + 1
    print('----------({}/{})开始处理图片{}----------'.format(count,num,imagepath))
    # print '人脸识别...'
    start = datetime.datetime.now()
    # c_faces = pre_crops_bydlib(imagepath)
    fs_align = faces_pre_path_alignment(imagepath)
    # fs_crop = faces_pre_crops_imgs_bydlib(fs_align)
    end = datetime.datetime.now()
    # print '共检测出{}张脸...'.format(len(c_faces))
    # print '人脸识别耗时{}秒'.format((end - start).seconds)
    count += 1
    # for f_crop in fs_crop:
    for f_crop in fs_align:
        try:
            f_z = face_pre_zoom(f_crop, 320, 320)
            time.sleep(0.1)
            f_g = face_pre_gray(f_z)
            clahe_face = face_pre_histogram_equalization(f_g)
            filepath, filename = os.path.split(imagepath)
            name, ext = os.path.splitext(filename)
            spath = os.path.join(savedir, name + '_%.f.jpg' % time.time())
            cv2.imwrite(spath, clahe_face)
            print('                                           ')
        except Exception as e:
            print(e)


# 多进程处理
def main(maxprocess, function, imagedir, savedir):
    imgspath = face_pre_get_image_path(imagedir)

    p = Pool(maxprocess)
    for imgpath in imgspath:
        p.apply_async(function, args=(imgpath, imagedir, savedir,))
    p.close()# 关闭进程池
    p.join()# 等待所有子进程完毕


if __name__ == '__main__':
    # t1 = datetime.datetime.now()
    # main(r'../test/ts5.png', r'../test/')
    # t2 = datetime.datetime.now()
    # print (t2-t1).seconds
    # pre_crops_bydlib(r'../test/faces.jpg')
    # faces_main(r'F:\image\imageData5\1000\*.jpg', r'F:\image\imageData5\1000p')
    # pre_crop_byopencv(r'C:\Users\Silance\PycharmProjects\faceDig\dataset\image\p\4-TS.JPG')

    imagedir = r'C:\Users\Silance\PycharmProjects\faceDig\dataset\image\test2'
    savedir = r'C:\Users\Silance\PycharmProjects\faceDig\dataset\image\test_2'

    path = r'C:\Users\Silance\PycharmProjects\faceDig\dataset\image\test2\20170810133434486.jpg'

    # main(4, faces_main, imagedir, savedir)
    # print face_alignment(path)

    faces = faces_pre_crops_path_bydlib(path)
    print(faces)

    imgs = faces_pre_imgs_alignment(faces[1])
    # print imgs

    # for img in imgs:
    #     detector = dlib.get_frontal_face_detector()
    #     faces = detector(img, 1)
    #     # print faces
    #     for face in faces:
    #         x1 = face.left()
    #         y1 = face.top()
    #         x2 = face.right()
    #         y2 = face.bottom()

    window = dlib.image_window()
    window.set_image(imgs)
    dlib.hit_enter_to_continue()


