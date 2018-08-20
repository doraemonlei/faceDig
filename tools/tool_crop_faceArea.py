#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/14 13:23
@author: Silence
截取脸部不同区域，包括眼睛长条区域，内眦赘皮邻域区域，前额区域，鼻梁区域，眼距区域
'''
import cv2
import numpy as np

import tool_get_landmarksAndImages
import tool_get_grey_image
from tool_blob_detection import get_blobs


def get_eyes_area(img):
    # 截取眼睛长条区域
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

    # 设置截取的矩形坐标
    x1 = left_x
    y1 = left_top_y

    x2 = right_x
    y2 = right_down_y

    # 比较两眼纵坐标，旋转后也需要判断
    if left_top_y > right_top_y:
        y1 = right_top_y
    elif left_down_y > right_down_y:
        y2 = left_down_y

    # 剪裁
    crop = img[y1-10:y2+10, x1:x2]

    # 置为统一尺寸
    eyes = cv2.resize(crop, (320, 50))

    return eyes


def get_epicanthus_area(img):
    # 截取内眦赘皮邻域区域
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)

    # 取出内眦处的两点（39，42）坐标
    left_x = landmarks[39].tolist()[0][0]
    left_y = landmarks[39].tolist()[0][1]

    right_x = landmarks[42].tolist()[0][0]
    right_y = landmarks[42].tolist()[0][1]

    # 扩展内眦两点坐标，取出内眦方形区域
    move = 10
    moveon = move*2
    x11 = left_x - move
    y11 = left_y - move
    x12 = x11 + moveon
    y12 = y11 + moveon

    x21 = right_x - move
    y21 = right_y - move
    x22 = x21 + moveon
    y22 = y21 + moveon

    epicanthus_area = [[x11, y11, x12, y12], [x21, y21, x22, y22]]

    return epicanthus_area


def get_forehead_area(img):
    # 截取前额区域
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)

    # 取出代表前额的两点（19，24）坐标
    left_x = landmarks[19].tolist()[0][0]
    left_y = landmarks[19].tolist()[0][1]

    right_x = landmarks[24].tolist()[0][0]
    right_y = landmarks[24].tolist()[0][1]

    # 扩展前额两点坐标，取出前额区域
    x1 = left_x
    # y1 = left_y - 50
    y1 = left_y - 25

    x2 = right_x
    y2 = right_y
    forehead_area = [x1, y1, x2, y2]

    return forehead_area


def get_nasalBridge_area(img):
    # 截取鼻梁区域
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)

    # 取出代表鼻梁的两点（29, 31, 33, 35）坐标
    up_x = landmarks[31].tolist()[0][0]
    up_y = landmarks[29].tolist()[0][1]

    down_x = landmarks[35].tolist()[0][0]
    down_y = landmarks[33].tolist()[0][1]

    # 扩展前额两点坐标，取出前额区域
    x1 = up_x
    y1 = up_y

    x2 = down_x
    y2 = down_y

    nasalBridge_area = [x1, y1, x2, y2]

    return nasalBridge_area


def get_ocular_area(img):
    # 眼距，用连线表示
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)

    # 取出内眦处的两点（39，42）坐标
    left_x = landmarks[39].tolist()[0][0]
    left_y = landmarks[39].tolist()[0][1]

    right_x = landmarks[42].tolist()[0][0]
    right_y = landmarks[42].tolist()[0][1]

    x1 = left_x
    y1 = left_y

    x2 = right_x
    y2 = right_y

    ocular_area = [x1, y1, x2, y2]

    return ocular_area

if __name__ == '__main__':
    # 测试

    # print get_eyes_area('../test/ts2.jpg')
    # get_epicanthus_area('../test/ts4.jpg')
    # get_forehead_area('../test/ts4.jpg')
    # get_nasalBridge_area('../test/ts4.jpg')
    # get_ocular_area('../test/ts4.jpg')

    path = r'../test/ts.jpg'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    epicanthus_area = get_epicanthus_area(img)
    forehead_area = get_forehead_area(img)
    nasalBridge_area = get_nasalBridge_area(img)
    ocular_area = get_ocular_area(img)
    blobs = get_blobs(img)

    cv2.line(img,(ocular_area[0],ocular_area[1]),(ocular_area[2],ocular_area[3]),color=(0,0,255),thickness=1)
    img = cv2.drawKeypoints(img, blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    listarr = [epicanthus_area[0],epicanthus_area[1],forehead_area,nasalBridge_area]
    for x1,y1,x2,y2 in listarr:
        cv2.rectangle(img, (x1,y1),(x2,y2),color=(0,0,255),thickness=1)

    cv2.imshow('1',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()