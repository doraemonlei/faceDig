#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/14 13:23
@author: Silence
截取眼睛区域
'''
import cv2
import tool_get_landmarksAndImages
import tool_get_grey_image

def get_eyes_area(img):

    img,landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)

    # 取出眼睛各点坐标
    left_x = landmarks[36].tolist()[0][0]
    left_y = landmarks[36].tolist()[0][1]
    left_top_x = landmarks[37].tolist()[0][0]
    left_top_y = landmarks[37].tolist()[0][1]
    left_down_x = landmarks[40].tolist()[0][0]
    left_down_y = landmarks[40].tolist()[0][1]

    right_x = landmarks[45].tolist()[0][0]
    right_y = landmarks[45].tolist()[0][1]
    right_top_x = landmarks[43].tolist()[0][0]
    right_top_y = landmarks[43].tolist()[0][1]
    right_down_x = landmarks[46].tolist()[0][0]
    right_down_y = landmarks[46].tolist()[0][1]

    #设置截取的矩形坐标
    x1 = left_x
    y1 = left_top_y

    x2 = right_x
    y2 = right_down_y

    #比较两眼纵坐标，旋转后也需要判断
    if left_top_y > right_top_y:
        y1 = right_top_y
    elif left_down_y > right_down_y:
        y2 = left_down_y
    crop = img[y1-10:y2+10,x1:x2]
    # img[x1:x2,y1-10:y2+10] = (0,0,255)
    # img[0:100,0:100] = (0,0,255)
    # print img
    # print img[0:100,0:100]
    # cv2.rectangle(img, (x1, y1-10), (x2, y2+10), (0, 0, 255), 2)
    # print crop.shape
    eyes = cv2.resize(crop,(320,50))
    # print eyes.shape
    # cv2.imshow('eye',eyes)
    # cv2.imshow('or',crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return eyes

def get_forehead_area():
    pass

def get_nasalBridge_area():
    pass

def get_ocular_area():
    pass

def get_melanocyticNevus_area():
    pass

if __name__ == '__main__':
    print get_eyes_area('../test/ts2.jpg')