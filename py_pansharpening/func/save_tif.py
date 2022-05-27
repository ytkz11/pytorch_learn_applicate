#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/27 16:29 
# @Author : DKY
# @File : save_tif.py
from osgeo import gdal
import numpy as np
import os


# 减少一行

def save_tif(file):
    def saveTif(array, cols, rows, driver, proj, Transform, band, filename):
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

        indexset = driver.Create(filename, cols+ 1, rows , band, gdal.GDT_Float32)
        indexset.SetGeoTransform(Transform)
        indexset.SetProjection(proj)
        if band>1:
            for i in range(band):
                Band = indexset.GetRasterBand(i + 1)
                Band.WriteArray(array[i, :, :], 0, 0)
        else:
            Band = indexset.GetRasterBand(1)
            Band.WriteArray(array[:, :], 0, 0)

    g_set = gdal.Open(file)

    # attribute
    transform = g_set.GetGeoTransform()
    driver = g_set.GetDriver()
    geoTransform = g_set.GetGeoTransform()
    ListgeoTransform = list(geoTransform)
    ListgeoTransform[5] = -ListgeoTransform[5]
    newgeoTransform = tuple(ListgeoTransform)
    proj = g_set.GetProjection()

    cols = g_set.RasterXSize
    rows = g_set.RasterYSize
    band = g_set.RasterCount
    # 读取tif
    index = g_set.ReadAsArray()

    index = np.c_[index, index[:, 0]]
    indexname = os.path.splitext(file)[0] + '_msi.tif'
    saveTif(index, cols, rows, driver, proj, newgeoTransform, band, indexname)


if __name__ == '__main__':
    file = r'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images\pan.tif'
    save_tif(file)
