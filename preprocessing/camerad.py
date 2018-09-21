#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/9/7 16:05
@author: Silence
'''
# coding:utf-8
'''
人脸位置校准
'''
import cv2
import dlib
from dataset import LANDMARKS_5_PATH

predictor_path = LANDMARKS_5_PATH
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("cannot open camear")
    exit(0)

while True:
    ret, frame = camera.read()

    if not ret:
        continue
    cv2.imshow('camera', frame)
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 检测脸部
    dets = detector(frame_new, 1)
    print("Number of faces detected: {}".format(len(dets)))
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in current frame")
        continue
    # 查找脸部位置
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(frame_new, detection))
    images = dlib.get_face_chips(frame_new, faces, size=320)
    for image in images:
        cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', cv_bgr_img)
    image = dlib.get_face_chip(frame_new, faces[0])
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', cv_bgr_img)

    k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    if k == ord('q'):  # 若检测到按键 ‘q’，退出
        break

cv2.destroyAllWindows()