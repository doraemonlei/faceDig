#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
人脸校正，剪裁
@time: 2017/6/23 15:33
@author: Silence
'''
# !/usr/bin/env python

import cv2
from tools import tool_get_grey_image
from tools import tool_face_detection_byopencv

def show_image(path):

    img = tool_get_grey_image.get_img(path)

    face_cascade = cv2.CascadeClassifier('../tools/haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    eye_cascade = cv2.CascadeClassifier('../tools/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    eyes = eye_cascade.detectMultiScale(img, 1.1, 5)
    for (x, y, w, h) in faces:
        crop = img[y:y + h, x:x + w]
        cv2.rectangle(crop, (x, y), (x + w, y + h), (255, 0, 255), 2)

        eyes = tool_face_detection_byopencv.eye_dect(img)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(crop, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def img_crop(path):

    img = tool_get_grey_image.get_img(path)
    face_cascade = cv2.CascadeClassifier('../tools/haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # faces = tool_face_detection.face_dect(img)
    i = 1
    for (x, y, w, h) in faces:
        n = path.split('/')[-1]
        print n
        crop = img[y:y + h, x:x + w]
        cv2.imwrite(('%s_' + n) % i, crop)
        i += 1

if __name__ == '__main__':

    # img_crop('../test/or/326.jpg')
    show_image('../test/or/326.jpg')