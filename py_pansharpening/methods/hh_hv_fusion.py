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
import cv2

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


def saveTif(array, col, row, driver, proj, Transform, filename, band=1):
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

    indexset = driver.Create(filename,col,  row, band, gdal.GDT_Float32)

    indexset.SetGeoTransform(Transform)
    indexset.SetProjection(proj)
    if band == 1:

        Band = indexset.GetRasterBand(1)
        Band.WriteArray(array, 0, 0)
    else:
        for i in range(band):
            Band = indexset.GetRasterBand(i + 1)
            Band.WriteArray(array[:, :, i])


def nearest(image, target_size):
    """
    Nearest Neighbour interpolate for RGB  image

    :param image: rgb image
    :param target_size: tuple = (height, width)
    :return: None
    """
    if target_size[0] < image.shape[0] or target_size[1] < image.shape[1]:
        raise ValueError("target image must bigger than input image")
    # 1：按照尺寸创建目标图像

    target_image = np.zeros(shape=(*target_size, 3))
    # 2:计算height和width的缩放因子
    alpha_h = target_size[0] / image.shape[0]
    alpha_w = target_size[1] / image.shape[1]

    for tar_x in range(target_image.shape[0] - 1):
        for tar_y in range(target_image.shape[1] - 1):
            # 3:计算目标图像人任一像素点
            # target_image[tar_x,tar_y]需要从原始图像
            # 的哪个确定的像素点image[src_x, xrc_y]取值
            # 也就是计算坐标的映射关系
            src_x = round(tar_x / alpha_h)
            src_y = round(tar_y / alpha_w)

            # 4：对目标图像的任一像素点赋值
            target_image[tar_x, tar_y] = image[src_x, src_y]

    return target_image
def main():
    hh = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/sar1_clip.tif'
    pan = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/pan1_Clip.tif'
    mss = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/mss1_Clip.tif'
    ds = gdal.Open(hh)
    pan_ds  = gdal.Open(pan)

    mss_ds  = gdal.Open(mss)

    rows = mss_ds.RasterYSize
    cols = mss_ds.RasterXSize


    original_1 = mss_ds.GetRasterBand(1).ReadAsArray()
    original_2 = mss_ds.GetRasterBand(2).ReadAsArray()
    original_3 = mss_ds.GetRasterBand(3).ReadAsArray()
    sar_1 = ds.GetRasterBand(1).ReadAsArray()
    sar_2 = ds.GetRasterBand(2).ReadAsArray()
    sar_3 = ds.GetRasterBand(3).ReadAsArray()
    used_1 = resize(sar_1, (int(rows/1), int(cols/1)))
    used_2 = nearest(sar_1, (int(rows/1), int(cols/1)))

    temp_arr = np.zeros(shape= (int(rows/1), int(cols/1), 4))
    temp_arr[:, :, 0] = original_1
    temp_arr[:, :, 1] = original_2
    temp_arr[:, :, 2] = used_1
    temp_arr[:, :, 3] = original_3


    filename = os.path.splitext(hh)[0] + '_resize_mss_hhhv.tif'
    geoTransform = mss_ds.GetGeoTransform()
    ListgeoTransform = list(geoTransform)
    newgeoTransform = tuple(ListgeoTransform)
    driver = mss_ds.GetDriver()
    proj = mss_ds.GetProjection()
    saveTif(temp_arr, int(cols/1), int(rows/1), driver, proj, newgeoTransform, filename, band=4)


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
    ListgeoTransform[5] = ListgeoTransform[5]
    newgeoTransform = tuple(ListgeoTransform)
    driver = hh_DataSet.GetDriver()
    proj = hh_DataSet.GetProjection()

    indexset = driver.Create(filename, cols, rows, 3, gdal.GDT_Int32)
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
    main()
    # combine_gf3()

    type = 0
