#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/20 16:18
@author: Silence
人脸检测通过dlib
'''
import cv2
import dlib
import matplotlib.pylab as plt
import tool_get_landmarksAndImages

# def crop_face(img):
#     img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)
#     '''
#     需求：
#     由于opencv检测人脸不太准确，有时候检测不出来。而且由于dlib检测人脸要根据68特征点
#     具体方法：
#
#     '''

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("D:\py2.7.12\Lib\site-packages\dlib-19.2.0-py2.7-win-amd64.egg\shape_predictor_68_face_landmarks.dat")


def geteye_rect(imgpath):
    bgrImg = cv2.imread(imgpath)
    if bgrImg is None:
        return False
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    facesrect = face_detector(rgbImg, 1)
    if len(facesrect) <= 0:
        return False

    for k, d in enumerate(facesrect):
        shape = landmark_predictor(rgbImg, d)
        for i in range(68):
            pt = shape.part(i)
            plt.plot(pt.x, pt.y, 'ro')
        plt.imshow(rgbImg)
        plt.show()


import cv2
import dlib
import numpy as np


# 根据人脸框bbox，从一张完整图片裁剪出人脸
def getface(path):
    bgrImg = cv2.imread(path,cv2.IMREAD_COLOR)
    print bgrImg.shape
    # rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    detector = dlib.get_frontal_face_detector()
    faces = detector(bgrImg, 1)
    for i,d in enumerate(faces):
        print i
        print d.left(), d.top(), d.right(), d.bottom()
    # dets, scores, idx = detector.run(bgrImg, 1)
    # for i, d in enumerate(dets):
    #     print("Detection {}, dets{},score: {}, face_type:{}".format(i, d, scores[i], idx[i]))
        # if len(faces) > 0:
    #     face = max(faces, key=lambda rect: rect.width() * rect.height())
    #     [x1, x2, y1, y2] = [face.left(), face.right(), face.top(), face.bottom()]

if __name__ == '__main__':
    # geteye_rect(r'F:\image\all\p\120-TS.jpg')
    getface(r'F:\image\all\p\120-TS.jpg')