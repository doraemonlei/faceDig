#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
摄像头实时检测人脸
@time: 2017/6/23 12:35
@author: Silence
'''

import cv2

def CatchUsbVideo(window_name):
    '''
    从摄像头实时检测人脸，并用方框框起来
    :param window_name: 显示窗口的名字
    :return: 检测窗口
    '''
    cv2.namedWindow(window_name)

    # 视频来源，0默认本地摄像头
    cap = cv2.VideoCapture(0)

    classfier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    color = (0, 255, 0)

    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    CatchUsbVideo()