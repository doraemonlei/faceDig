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


def load_facedata(return_X_y=False):

    module_path = dirname(__file__)
    # print module_path
    with open(join(module_path, 'facedata.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        print temp
        n_samples = int(temp[6])
        n_features = int(temp[7])
        target_names = np.array(temp[0:5])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:5], dtype=np.float64)
            target[i] = np.asarray(ir[5], dtype=np.int)

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=['forehead_feature', 'nasalBridge_feature',
                                'ocular_feature', 'melanocyticNevus_feature',
                                'epicanthus_feature'])


if __name__ == '__main__':
    face = load_facedata()
    print face.feature_names