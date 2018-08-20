#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/9/30 13:00
@author: Silence
'''
import os
import glob
import dataset

path = dataset.POSITIVE_IMAGE
print len(glob.glob(path))