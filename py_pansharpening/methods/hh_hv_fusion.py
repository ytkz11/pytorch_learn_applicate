#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/30 14:38 
# @Author : DKY
# @File : hh_hv_fusion.py
from osgeo import gdal
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
import os


def read_tif_to_np(file):
    msi = gdal.Open(file)

    original_msi1 = msi.ReadAsArray()

    shape = np.shape(original_msi1)

    if len(shape) > 2:
        z, x, y = shape[0], shape[1], shape[2]
        temp = np.zeros(shape=(x, y, z))
        for i in range(z):
            temp[:, :, i] = original_msi1[i, :, :]
    else:
        x, y = shape[0], shape[1]
        temp = np.zeros(shape=(x, y))
        temp[:, :] = original_msi1[:, :]
    return temp


def saveTif(array, cols, rows, driver, proj, Transform, filename, band=1):
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
    if band == 1:

        Band = indexset.GetRasterBand(1)
        Band.WriteArray(array, 0, 0)
    else:
        for i in range(band):
            Band = indexset.GetRasterBand(i + 1)
            Band.WriteArray(array[:, :, i])


def main():
    hh = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/GF3HH13_Clip.tif'
    hv = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/hv1_Clip.tif'
    ds = gdal.Open(hv)
    original_hh = read_tif_to_np(hh)
    original_hv = read_tif_to_np(hv)
    x, y = original_hv.shape
    used_hh = resize(original_hh, (x, y))
    original_hv_hh = (used_hh ** 2 + original_hv ** 2) ** 0.5 / 2
    plt.imshow(original_hv_hh), plt.show()
    temp_arr = np.zeros(shape=(x, y, 3))

    temp_arr[:, :, 0] = original_hv_hh
    temp_arr[:, :, 1] = used_hh
    temp_arr[:, :, 2] = original_hv
    cols, rows = y, x
    filename = os.path.splitext(hv)[0] + 'sar.tif'
    geoTransform = ds.GetGeoTransform()
    ListgeoTransform = list(geoTransform)
    ListgeoTransform[5] = -ListgeoTransform[5]
    newgeoTransform = tuple(ListgeoTransform)
    driver = ds.GetDriver()
    proj = ds.GetProjection()
    saveTif(temp_arr, cols, rows, driver, proj, newgeoTransform, filename, band=3)


def combine_gf3():
    hh = 'X:\DengKaiYuan\henan\GF3\GF3_SAY_FSII_025440_E113.7_N35.1_20210609_L2_HHHV_L20005692256\GF3_SAY_FSII_025440_E113.7_N35.1_20210609_L2_HH_L20005692256.tiff'
    hv = 'X:\DengKaiYuan\henan\GF3\GF3_SAY_FSII_025440_E113.7_N35.1_20210609_L2_HHHV_L20005692256\GF3_SAY_FSII_025440_E113.7_N35.1_20210609_L2_HV_L20005692256.tiff'

    hh_DataSet = gdal.Open(hh)
    hv_DataSet = gdal.Open(hv)

    rows = hv_DataSet.RasterYSize
    cols = hv_DataSet.RasterXSize
    filename = os.path.splitext(hv)[0] + '_combined.tif'
    geoTransform = hh_DataSet.GetGeoTransform()
    ListgeoTransform = list(geoTransform)
    ListgeoTransform[5] = -ListgeoTransform[5]
    newgeoTransform = tuple(ListgeoTransform)
    driver = hh_DataSet.GetDriver()
    proj = hh_DataSet.GetProjection()

    indexset = driver.Create(filename, cols, rows, 3, gdal.GDT_Float32)
    indexset.SetGeoTransform(newgeoTransform)
    indexset.SetProjection(proj)
    # 分块
    nBlockSize = 500
    i = 0
    j = 0
    try:
        while i < rows:
            while j < cols:
                # 保存分块大小
                nXBK = nBlockSize
                nYBK = nBlockSize
                if i + nBlockSize > rows:
                    nYBK = rows - i
                if j + nBlockSize > cols:
                    nXBK = cols - j
                hh = hh_DataSet.ReadAsArray(j, i, nXBK, nYBK)
                hv = hv_DataSet.ReadAsArray(j, i, nXBK, nYBK)
                hh_hv = (hh ** 2 + hv ** 2) ** 0.5
                print(i)
                for m in range(3):
                    ReadBand = indexset.GetRasterBand(m + 1)

                    ReadBand.SetNoDataValue(-9999)
                    if m == 0:
                        ReadBand.WriteArray(hh_hv, j, i)
                    if m == 1:
                        ReadBand.WriteArray(hh, j, i)
                    if m == 2:
                        ReadBand.WriteArray(hv, j, i)
                del hh_hv, hh, hv

                j = j + nXBK

            j = 0
            i = i + nYBK
    except KeyboardInterrupt:

        raise


if __name__ == '__main__':
    # main()
    combine_gf3()

    type = 0