#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2018/3/1 12:04
@author: Silence

标记68特征点，将标记的图片进行保存
输入图片路径即可

如：
main(r'../test/12122_zoom_gray.jpg')
main(r'../test/*.jpg')
'''
import glob
import cv2

from tools.tool_get_landmarksAndImages import get_landmarks_m

def main(path):

    for path in glob.glob(path):
        a, b = path.split('.')
        fileneme = a + '_mark.' + b
        img = cv2.imread(path)
        img = get_landmarks_m(img)
        cv2.imwrite(fileneme, img)

if __name__ == '__main__':
    main(r'C:\Users\Silance\Desktop\12122_zoom_gray.jpg')