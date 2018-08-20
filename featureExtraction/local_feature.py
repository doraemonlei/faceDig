#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/13 15:40
@author: Silence
获取5个局部特征
'''
from tools import tool_get_landmarksAndImages
from tools import tool_get_grey_image
from tools import tool_blob_detection
from tools import tool_get_gabor_feature
import numpy as np
import glob
import os


def get_forehead_feature(img):
    img,landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)
    forehead_distance = np.linalg.norm(landmarks[26] - landmarks[17])
    return forehead_distance


def get_melanocyticNevus_feature(img):
    melanocyticNevus_num = tool_blob_detection.blob_detection(img)
    return melanocyticNevus_num


def get_epicanthus_feature(img):
    eye_feature = tool_get_gabor_feature.get_gabor_feature(img)
    return eye_feature


def get_nasalBridge_feature(img):
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)
    nasalBridge_distance = np.linalg.norm(landmarks[33] - landmarks[30])
    return nasalBridge_distance


def get_ocular_feature(img):
    img, landmarks = tool_get_landmarksAndImages.read_im_and_landmarks(img)
    dis1 = np.linalg.norm(landmarks[42] - landmarks[39])
    dis2 = np.linalg.norm(landmarks[45] - landmarks[36])
    ocular_distance = dis1/dis2
    return ocular_distance

def main(path,target=0):
    np_list = []
    for path in glob.glob(path):
        print path
        img = tool_get_grey_image.get_img(path)
        v1 = round(get_forehead_feature(img), 2)
        v2 = round(get_nasalBridge_feature(img), 2)
        v3 = round(get_ocular_feature(img), 2)
        v4 = get_melanocyticNevus_feature(img)
        v5 = round(get_epicanthus_feature(img), 2)
        v6 = target
        local_list = [v1,v2,v3,v4,v5,v6]
        np_list.append(local_list)
    return np_list

def main2(path):
    img = tool_get_grey_image.get_img(path)
    v1 = round(get_forehead_feature(img), 2)
    v2 = round(get_nasalBridge_feature(img), 2)
    v3 = round(get_ocular_feature(img), 2)
    v4 = get_melanocyticNevus_feature(img)
    v5 = round(get_epicanthus_feature(img), 2)
    local_list = [v1,v2,v3,v4,v5]
    return local_list
if __name__ == '__main__':
    # print main(r'I:\image\chidren_frontface\tp\*.jpg')
    print main2(r'C:\Users\Silance\Desktop/12122_zoom_gray.jpg')