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
import dataset
from matplotlib import pyplot as plt

PREDICTOR_PATH = dataset.LANDMARKS_5_PATH
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
    # print("Number of faces detected: {}".format(len(rects)))

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
        cv2.circle(im, pos, 0, color=(0, 255, 255),thickness=-1)

    return im


#多张脸使用的一个例子
def get_landmarks_m(im):

    dets = detector(im, 1)
    #脸的个数
    print("脸的个数:%s" % len(dets))
    for i in range(len(dets)):
        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])
        for j in range(68):
            # 编号
            cv2.putText(im, str(j), (facepoint[j][0],facepoint[j][1]),
                        # fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.6,
                        color=(255, 255, 255))
            # 标记点
            cv2.circle(im, (facepoint[j][0],facepoint[j][1]), 3, color=(0, 255, 255),thickness=-1)
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

def show_image(imagepath):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dataset.LANDMARKS_68_PATH)

    # cv2读取图像
    img = cv2.imread(imagepath)

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])

        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])

            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(img, pos, 5, color=(0, 255, 0))

            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img)
    # plt.show()

    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    window = dlib.image_window()
    window.set_image(img)
    dlib.hit_enter_to_continue()



if __name__ == '__main__':
    # path = r'C:\Users\Silance\Desktop\12122_zoom_gray.jpg'
    # n = 1
    # for path in glob.glob(path):
    #     img = cv2.imread(path)
    #     img = get_landmarks_m(img)
    #     cv2.imwrite('%s.jpg'%n, img)
    #     n += 1


    path = r'C:\Users\Silance\PycharmProjects\faceDig\test\20171019133653792.jpg'
    show_image(path)



