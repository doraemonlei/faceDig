#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@time: 2017/7/10 16:13
@author: Silence
使用skimage中的gabor
'''

from skimage.filters import gabor,gabor_kernel
from skimage import data, io
from matplotlib import pyplot as plt

def get_gabor():
    '''
    调用skimage中的函数创建gabor滤波器
    :return:
    '''
    # image = data.coins()
    image = io.imread(r'../test/2.jpg', as_grey=True)
    print image
    # detecting edges in a coin image
    filt_real, filt_imag = gabor(image, frequency=0.6,n_stds=7,theta=180)
    plt.figure()
    io.imshow(filt_real)
    io.show()

    # less sensitivity to finer details with the lower frequency kernel
    filt_real, filt_imag = gabor(image, frequency=0.1)
    plt.figure()
    io.imshow(filt_real)
    # io.imshow(filt_imag)
    io.show()

def get_gabor_kernel():
    '''
    调用skimage中的函数创建gabor核
    :return:
    '''
    gk = gabor_kernel(frequency=0.2)
    plt.figure()
    io.imshow(gk.real)
    io.show()
    # more ripples (equivalent to increasing the size of the
    # Gaussian spread)
    gk = gabor_kernel(frequency=0.2, bandwidth=0.1)
    plt.figure()
    io.imshow(gk.real)
    io.show()

if __name__ == '__main__':
    get_gabor()
    get_gabor_kernel()