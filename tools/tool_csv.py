#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/21 14:57
@author: Silence
保存训练数据
'''
import csv
from featureExtraction import local_feature
import glob
def input_csv(imgpath):
    face_data = local_feature.main(imgpath,target=1)
    with open('facedata.csv', "wb") as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['forehead_feature','nasalBridge_feature',
                         'ocular_feature','melanocyticNevus_feature',
                         'epicanthus_feature','target'])
        for data in face_data:
            writer.writerow(data)

def ainput_csv(imgpath):
    face_data = local_feature.main(imgpath, target=0)
    with open('facedata.csv', "ab") as f:
        writer = csv.writer(f, delimiter=',')
        for data in face_data:
            writer.writerow(data)

if __name__ == '__main__':
    ainput_csv(r'F:\image\all\pp\n\*.jpg')