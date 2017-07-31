#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/5/18 15:35
@author: Silence
人脸色素痣检测
'''
import cv2
import numpy as np
import tool_get_grey_image as gi


def blob_detection(img):
    # 初始化参数
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 15
    params.maxArea = 500
    params.blobColor = 000
    params.minCircularity = 10
    params.maxCircularity = 10000
    detector = cv2.SimpleBlobDetector(params)

    # 检测斑点
    blobs = detector.detect(img)
    blobs_num = len(blobs)
    return blobs_num

    # for b in blobs:
    #     print b.pt
    #     print b.size
    #     print b.angle
    #     print b.class_id
    #     print b.octave
    #     print b.response


def show_blob_image(path):
    img = gi.get_img(path)

    # 初始化参数
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 15
    params.maxArea = 500
    params.blobColor = 000
    params.minCircularity = 10
    params.maxCircularity = 10000
    detector = cv2.SimpleBlobDetector(params)

    # 检测斑点
    blobs = detector.detect(img)
    # 画出斑点
    im_with_keypoints = cv2.drawKeypoints(img, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_blobs(img):
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 15
    params.maxArea = 500
    params.blobColor = 000
    params.minCircularity = 10
    params.maxCircularity = 10000
    detector = cv2.SimpleBlobDetector(params)

    # 检测斑点
    blobs = detector.detect(img)
    return blobs

'''
c源码，用于调整参数
SimpleBlobDetector::Params::Params()  
{  
    thresholdStep = 10;    //二值化的阈值步长，即公式1的t  
    minThreshold = 50;   //二值化的起始阈值，即公式1的T1  
    maxThreshold = 220;    //二值化的终止阈值，即公式1的T2  
    //重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点  
    minRepeatability = 2;     
    //最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点  
    minDistBetweenBlobs = 10;  
  
    filterByColor = true;    //斑点颜色的限制变量  
    blobColor = 0;    //表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点  
  
    filterByArea = true;    //斑点面积的限制变量  
    minArea = 25;    //斑点的最小面积  
    maxArea = 5000;    //斑点的最大面积  
  
    filterByCircularity = false;    //斑点圆度的限制变量，默认是不限制  
    minCircularity = 0.8f;    //斑点的最小圆度  
    //斑点的最大圆度，所能表示的float类型的最大值  
    maxCircularity = std::numeric_limits<float>::max();  
  
    filterByInertia = true;    //斑点惯性率的限制变量  
    //minInertiaRatio = 0.6;  
    minInertiaRatio = 0.1f;    //斑点的最小惯性率  
    maxInertiaRatio = std::numeric_limits<float>::max();    //斑点的最大惯性率  
  
    filterByConvexity = true;    //斑点凸度的限制变量  
    //minConvexity = 0.8;  
    minConvexity = 0.95f;    //斑点的最小凸度  
    maxConvexity = std::numeric_limits<float>::max();    //斑点的最大凸度  
}  
'''

if __name__ == '__main__':
    # print blob_detection('../test/ts3.jpg')
    show_blob_image('../test/ts2.jpg')