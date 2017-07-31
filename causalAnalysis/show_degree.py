#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@author: Silence
'''
import numpy as np
import cv2

from tools.tool_facedata import load_facedata
from tools.tool_crop_eyesArea import get_forehead_area
from tools.tool_crop_eyesArea import get_nasalBridge_area
from tools.tool_crop_eyesArea import get_ocular_area
from tools.tool_crop_eyesArea import get_epicanthus_area
from tools.tool_blob_detection import get_blobs


class Bunch(dict):

    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

def get_degree():
    fd = load_facedata()
    epicanthus_feature_samples = fd.epicanthus_feature[:73]
    forehead_feature_samples = fd.forehead_feature[0][:73]
    nasalBridge_feature_samples = fd.nasalBridge_feature[:73]
    ocular_feature_samples = fd.ocular_feature[:73]

    epicanthus_feature_degree = []
    forehead_feature_degree =[]
    nasalBridge_feature_degree = []
    ocular_feature_degree =[]
    for i in xrange(20, 100, 20):
        epicanthus_feature_degree.append(round(np.percentile(epicanthus_feature_samples, i), 2))
        forehead_feature_degree.append(round(np.percentile(forehead_feature_samples, i), 2))
        nasalBridge_feature_degree.append(round(np.percentile(nasalBridge_feature_samples, i), 2))
        ocular_feature_degree.append(round(np.percentile(ocular_feature_samples, i), 2))

    return Bunch(
        epicanthus_feature_degree=epicanthus_feature_degree,
        forehead_feature_degree=forehead_feature_degree,
        nasalBridge_feature_degree=nasalBridge_feature_degree,
        ocular_feature_degree=ocular_feature_degree
    )


def show_degree(path, sample):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    gd = get_degree()
    forehead_feature_degree = gd.forehead_feature_degree,
    nasalBridge_feature_degree = gd.nasalBridge_feature_degree,
    ocular_feature_degree = gd.ocular_feature_degree
    epicanthus_feature_degree = gd.epicanthus_feature_degree

    degree_0_green = (0, 255, 0)
    degree_1_blue = (255, 0, 0)
    degree_2_yellow = (0, 255, 255)
    degree_3_orange = (0, 165, 255)
    degree_4_red = (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1

    # 前额部分
    forehead_area = get_forehead_area(img)
    if sample[0] <= forehead_feature_degree[0]:
        cv2.rectangle(
            img,
            (forehead_area[0],forehead_area[1]), (forehead_area[2],forehead_area[3]),
            color=degree_1_blue,
            thickness=-1
        )
        cv2.putText(
            img,
            '1',
            ((forehead_area[0]+forehead_area[2])/2, (forehead_area[1]+forehead_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif forehead_feature_degree[0] < sample[0] <= forehead_feature_degree[1]:
        cv2.rectangle(
            img,
            (forehead_area[0],forehead_area[1]), (forehead_area[2],forehead_area[3]),
            color=degree_0_green,
            thickness=-1
        )
        cv2.putText(
            img,
            '0',
            ((forehead_area[0]+forehead_area[2])/2, (forehead_area[1]+forehead_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif forehead_feature_degree[1] < sample[0] <= forehead_feature_degree[2]:
        cv2.rectangle(
            img,
            (forehead_area[0],forehead_area[1]), (forehead_area[2],forehead_area[3]),
            color=degree_2_yellow,
            thickness=-1
        )
        cv2.putText(
            img,
            '2',
            ((forehead_area[0]+forehead_area[2])/2, (forehead_area[1]+forehead_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif forehead_feature_degree[2] < sample[0] <= forehead_feature_degree[3]:
        cv2.rectangle(
            img,
            (forehead_area[0],forehead_area[1]), (forehead_area[2],forehead_area[3]),
            color=degree_3_orange,
            thickness=-1
        )
        cv2.putText(
            img,
            '3',
            ((forehead_area[0]+forehead_area[2])/2, (forehead_area[1]+forehead_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    else:
        cv2.rectangle(
            img,
            (forehead_area[0],forehead_area[1]), (forehead_area[2],forehead_area[3]),
            color=degree_4_red,
            thickness=-1
        )
        cv2.putText(
            img,
            '4',
            ((forehead_area[0]+forehead_area[2])/2, (forehead_area[1]+forehead_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )

    # 鼻梁部分
    nasalBridge_area = get_nasalBridge_area(img)
    if sample[1] <= nasalBridge_feature_degree[0]:
        cv2.rectangle(
            img,
            (nasalBridge_area[0],nasalBridge_area[1]), (nasalBridge_area[2],nasalBridge_area[3]),
            color=degree_1_blue,
            thickness=-1
        )
        cv2.putText(
            img,
            '1',
            ((nasalBridge_area[0]+nasalBridge_area[2])/2, (nasalBridge_area[1]+nasalBridge_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif nasalBridge_feature_degree[0] < sample[1] <= nasalBridge_feature_degree[1]:
        cv2.rectangle(
            img,
            (nasalBridge_area[0],nasalBridge_area[1]), (nasalBridge_area[2],nasalBridge_area[3]),
            color=degree_0_green,
            thickness=-1
        )
        cv2.putText(
            img,
            '0',
            ((nasalBridge_area[0]+nasalBridge_area[2])/2, (nasalBridge_area[1]+nasalBridge_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif nasalBridge_feature_degree[1] < sample[1] <= nasalBridge_feature_degree[2]:
        cv2.rectangle(
            img,
            (nasalBridge_area[0],nasalBridge_area[1]), (nasalBridge_area[2],nasalBridge_area[3]),
            color=degree_2_yellow,
            thickness=-1
        )
        cv2.putText(
            img,
            '2',
            ((nasalBridge_area[0]+nasalBridge_area[2])/2, (nasalBridge_area[1]+nasalBridge_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif nasalBridge_feature_degree[2] < sample[1] <= nasalBridge_feature_degree[3]:
        cv2.rectangle(
            img,
            (nasalBridge_area[0],nasalBridge_area[1]), (nasalBridge_area[2],nasalBridge_area[3]),
            color=degree_3_orange,
            thickness=-1
        )
        cv2.putText(
            img,
            '3',
            ((nasalBridge_area[0]+nasalBridge_area[2])/2, (nasalBridge_area[1]+nasalBridge_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    else:
        cv2.rectangle(
            img,
            (nasalBridge_area[0],nasalBridge_area[1]), (nasalBridge_area[2],nasalBridge_area[3]),
            color=degree_4_red,
            thickness=-1
        )
        cv2.putText(
            img,
            '4',
            ((nasalBridge_area[0]+nasalBridge_area[2])/2, (nasalBridge_area[1]+nasalBridge_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )

    # 眼距部分
    ocular_area = get_ocular_area(img)
    if sample[2] <= ocular_feature_degree[0]:
        cv2.line(img,
                 (ocular_area[0], ocular_area[1]), (ocular_area[2], ocular_area[3]),
                 color=(0,0,255),
                 thickness=1
        )
        cv2.putText(
            img,
            '1',
            ((ocular_area[0]+ocular_area[2])/2, (ocular_area[1]+ocular_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif ocular_feature_degree[0] < sample[1] <= ocular_feature_degree[1]:
        cv2.line(img,
                 (ocular_area[0], ocular_area[1]), (ocular_area[2], ocular_area[3]),
                 color=(0, 0, 255),
                 thickness=1
        )
        cv2.putText(
            img,
            '0',
            ((ocular_area[0]+ocular_area[2])/2, (ocular_area[1]+ocular_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif ocular_feature_degree[1] < sample[1] <= ocular_feature_degree[2]:
        cv2.line(img,
                 (ocular_area[0], ocular_area[1]), (ocular_area[2], ocular_area[3]),
                 color=(0, 0, 255),
                 thickness=1
        )
        cv2.putText(
            img,
            '2',
            ((ocular_area[0]+ocular_area[2])/2, (ocular_area[1]+ocular_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    elif ocular_feature_degree[2] < sample[1] <= ocular_feature_degree[3]:
        cv2.line(img,
                 (ocular_area[0], ocular_area[1]), (ocular_area[2], ocular_area[3]),
                 color=(0, 0, 255),
                 thickness=1
        )
        cv2.putText(
            img,
            '3',
            ((ocular_area[0]+ocular_area[2])/2, (ocular_area[1]+ocular_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )
    else:
        cv2.line(img,
                 (ocular_area[0], ocular_area[1]), (ocular_area[2], ocular_area[3]),
                 color=(0, 0, 255),
                 thickness=1
        )
        cv2.putText(
            img,
            '4',
            ((ocular_area[0]+ocular_area[2])/2, (ocular_area[1]+ocular_area[3])/2),
            font,
            fontScale=font_scale,
            color=font_color,
            thickness=font_thickness
        )

    # 内眦部分
    epicanthus_area = get_epicanthus_area(img)
    if sample[4] <= epicanthus_feature_degree[0]:
        for x1, y1, x2, y2 in epicanthus_area:
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=degree_1_blue,
                thickness=-1
            )
            cv2.putText(
                img,
                '1',
                ((x1+x2)/2, (y1+y2)/2),
                font,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness
            )
    elif epicanthus_feature_degree[0] < sample[4] <= epicanthus_feature_degree[1]:
        for x1, y1, x2, y2 in epicanthus_area:
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=degree_1_blue,
                thickness=-1
            )
            cv2.putText(
                img,
                '1',
                ((x1+x2)/2, (y1+y2)/2),
                font,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness
            )
    elif epicanthus_feature_degree[1] < sample[4] <= epicanthus_feature_degree[2]:
        for x1, y1, x2, y2 in epicanthus_area:
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=degree_1_blue,
                thickness=-1
            )
            cv2.putText(
                img,
                '1',
                ((x1+x2)/2, (y1+y2)/2),
                font,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness
            )
    elif epicanthus_feature_degree[2] < sample[4] <= epicanthus_feature_degree[3]:
        for x1, y1, x2, y2 in epicanthus_area:
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=degree_1_blue,
                thickness=-1
            )
            cv2.putText(
                img,
                '1',
                ((x1+x2)/2, (y1+y2)/2),
                font,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness
            )
    else:
        for x1, y1, x2, y2 in epicanthus_area:
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=degree_1_blue,
                thickness=-1
            )
            cv2.putText(
                img,
                '1',
                ((x1+x2)/2, (y1+y2)/2),
                font,
                fontScale=font_scale,
                color=font_color,
                thickness=font_thickness
            )

    # 显示黑痣
    blobs = get_blobs(img)
    img = cv2.drawKeypoints(img, blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    path = path = r'../test/ts.jpg'
    sample = [442.37, 57.22, 0.44, 12, 67.04]
    show_degree(path, sample)