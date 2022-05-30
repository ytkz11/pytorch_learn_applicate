#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/30 14:38 
# @Author : DKY
# @File : hh_hv_fusion.py
from osgeo import gdal
import numpy as np
from h5_model import read_tif_to_np
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
def saveTif(array, cols, rows, driver, proj, Transform, filename,band =1):
    '''

    @param array: 数据矩阵
    @param cols: 列
    @param rows: 行
    @param driver:
    @param proj: 投影
    @param Transform: 存储着栅格数据集的地理坐标信息
    @param filename: 输出文件名字
    @return:
    '''

    indexset = driver.Create(filename, cols, rows, band, gdal.GDT_Float32)
    indexset.SetGeoTransform(Transform)
    indexset.SetProjection(proj)
    if band ==1:

        Band = indexset.GetRasterBand(1)
        Band.WriteArray(array, 0, 0)
    else:
        for i in range(band):
            Band = indexset.GetRasterBand(i+1)
            Band.WriteArray(array[:,:,i], 0, 0)

def main():
    hh = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/GF3HH13_Clip.tif'
    hv = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/hv1_Clip.tif'
    ds = gdal.Open(hv)
    original_hh = read_tif_to_np(hh)
    original_hv = read_tif_to_np(hv)
    x, y = original_hv.shape
    used_hh = resize(original_hh, (x, y))
    original_hv_hh = (used_hh ** 2 + original_hv ** 2) ** 0.5
    plt.imshow(original_hv_hh), plt.show()
    temp_arr = np.zeros(shape=(x, y, 3))

    temp_arr[:, :, 0] = original_hv_hh
    temp_arr[:, :, 0] = used_hh
    temp_arr[:, :, 0] = original_hv
    cols, rows =x, y
    filename = os.path.splitext(hv)[0] + 'sar.tif'
    geoTransform = ds.GetGeoTransform()
    ListgeoTransform = list(geoTransform)
    ListgeoTransform[5] = -ListgeoTransform[5]
    newgeoTransform = tuple(ListgeoTransform)
    driver = ds.GetDriver()
    proj = ds.GetProjection()
    saveTif(temp_arr, cols, rows, driver, proj, newgeoTransform, filename,band=3)
if __name__ == '__main__':
    main()

    type =0



