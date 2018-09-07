#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/9/27 16:36
@author: Silence
'''
import os

HAARCASCADES_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'haarcascades/haarcascade_frontalface_alt_tree.xml'
    )
)

LANDMARKS_68_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'shape_predictor_68_face_landmarks.dat'
    )
)

LANDMARKS_5_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'shape_predictor_5_face_landmarks.dat'
    )
)

FACEDATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'facedata.csv'
    )
)

IMAGE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'image'
    )
)

POSITIVE_IMAGE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'image',
        'p',
        '*.jpg'
    )
)

NEGATIVE_IMAGE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'image',
        'n',
        '*.jpg'
    )
)

POSITIVE_REC_IMAGE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'image',
        'pr',
    )
)

NEGATIVE_REC_IMAGE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        'image',
        'nr',
    )
)

if __name__ == '__main__':
    print LANDMARKS_5_PATH