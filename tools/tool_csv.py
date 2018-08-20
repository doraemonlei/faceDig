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
def input_csv(imgpath,filepath,sam_num):
    face_data = local_feature.main(imgpath,target=1)
    with open(filepath, "wb") as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['forehead_feature','nasalBridge_feature',
                         'ocular_feature','melanocyticNevus_feature',
                         'epicanthus_feature','target',sam_num,5])
        for data in face_data:
            writer.writerow(data)

def ainput_csv(imgpath,filepath):
    face_data = local_feature.main(imgpath, target=0)
    with open(filepath, "ab") as f:
        writer = csv.writer(f, delimiter=',')
        for data in face_data:
            writer.writerow(data)

if __name__ == '__main__':
    input_csv(r'I:\image\chidren_frontface\tp\*.jpg')
    # ainput_csv(r'F:\image\8.17\test\n\*.jpg')