#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/21 15:47
@author: Silence
获取人脸训练数据
'''

import csv
from os.path import dirname
from os.path import join
import numpy as np

import dataset

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


def load_facedata(csv_path=dataset.FACEDATA_PATH, return_X_y=False):

    module_path = dirname(__file__)
    # print module_path
    with open(join(module_path, csv_path)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        # print temp
        n_samples = int(temp[6])
        n_features = int(temp[7])
        target_names = np.array(temp[0:5])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        forehead_feature = np.empty((n_samples, 1))
        nasalBridge_feature = np.empty((n_samples, 1))
        ocular_feature = np.empty((n_samples, 1))
        melanocyticNevus_feature = np.empty((n_samples, 1))
        epicanthus_feature = np.empty((n_samples, 1))

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:5], dtype=np.float64)
            target[i] = np.asarray(ir[5], dtype=np.int)
            forehead_feature[i] = np.asarray(ir[0], dtype=np.float64)
            nasalBridge_feature[i] = np.asarray(ir[1], dtype=np.float64)
            ocular_feature[i] = np.asarray(ir[2], dtype=np.float64)
            melanocyticNevus_feature[i] = np.asarray(ir[3], dtype=np.float64)
            epicanthus_feature[i] = np.asarray(ir[4], dtype=np.float64)

        forehead_feature = forehead_feature.reshape(1,n_samples)

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        target_names=target_names,
        forehead_feature=forehead_feature,
        nasalBridge_feature=nasalBridge_feature,
        ocular_feature=ocular_feature,
        melanocyticNevus_feature=melanocyticNevus_feature,
        epicanthus_feature=epicanthus_feature,
        feature_names=['forehead_feature', 'nasalBridge_feature',
                       'ocular_feature', 'melanocyticNevus_feature',
                       'epicanthus_feature']
    )


if __name__ == '__main__':
    face = load_facedata()

    print face.forehead_feature[0][:73]
    print np.max(face.forehead_feature[0][:73])
    print np.min(face.forehead_feature[0][:73])

    # 均值
    print np.mean(face.forehead_feature[0][:73])
    # 中值
    print np.median(face.forehead_feature[0][:73])
    # 方差
    print np.var(face.forehead_feature[0][:73])