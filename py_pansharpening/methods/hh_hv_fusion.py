#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/30 14:38 
# @Author : DKY
# @File : hh_hv_fusion.py
from osgeo import gdal
import numpy as np
from utils import upsample_interp23, downgrade_images
import random
from tqdm import tqdm
import cv2
from h5_model import read_tif_to_np
from skimage.transform import resize
import matplotlib.pyplot as plt



if __name__ == '__main__':
    hh = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/GF3HH13_Clip.tif'
    hv = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/hv1_Clip.tif'

    original_hh = read_tif_to_np(hh)
    original_hv = read_tif_to_np(hv)
    x,y =original_hv.shape
    used_hh = resize(original_hh, (x, y))
    original_hv_hh = (used_hh**2+original_hv**2)**0.5
    plt.imshow(original_hv_hh), plt.show()

    type =0



