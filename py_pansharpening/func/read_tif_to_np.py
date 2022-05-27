#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/27 15:33 
# @Author : DKY
# @File : read_tif_to_np.py
from osgeo import gdal
import numpy as np

def read_tif_to_np(file):
    msi = gdal.Open(file)

    original_msi1 = msi.ReadAsArray()

    shape=np.shape(original_msi1)

    if len(shape) >2:
        z, x, y =shape[0],shape[1],shape[2]
        temp = np.zeros(shape=(x, y, z))
        for i in range(z):
            temp[:,:,i] = original_msi1[i,:,:]
    else:
        x, y = shape[0], shape[1]
        temp = np.zeros(shape=(x, y))
        temp[:, :] = original_msi1[:, :]
    return temp

if __name__ == '__main__':
    file = r'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images\pms_Clip1.tif'
    t = read_tif_to_np(file)
    a =0