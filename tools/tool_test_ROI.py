#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/12 16:38
@author: Silence
测试ROI
'''

# 文件说明:
# 利用OpenCv的ROI区域,将衣服图片切成规则的小图片的python程序

import cv2
import numpy as np

image_path = "../test/cat.png"
srcImg = cv2.imread(image_path)
cv2.namedWindow("[srcImg]", cv2.WINDOW_AUTOSIZE)
cv2.imshow("[srcImg]", srcImg)

image_save_path_head = "../test/cat/cat_ROI_"
image_save_path_tail = ".jpg"
seq = 1
for i in range(15):
    for j in range(11):
        img_roi = srcImg[(i * 32):((i + 1) * 32), (j * 32):((j + 1) * 32)];

        cv2.namedWindow("[ROI_Img]", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("[ROI_Img]", img_roi)
        cv2.waitKey(500)
        cv2.destroyWindow("[ROI_Img]")
        image_save_path = "%s%d%s" % (image_save_path_head, seq, image_save_path_tail)
        cv2.imwrite(image_save_path, img_roi)
        seq = seq + 1
cv2.waitKey(0)
cv2.destroyWindow("[srcImg]")