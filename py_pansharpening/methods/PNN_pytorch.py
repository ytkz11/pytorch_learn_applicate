#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/30 10:08 
# @Author : DKY
# @File : PNN_pytorch.py
from osgeo import gdal
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
from torch.autograd import Variable
import torch.utils.data as Data

def pnn_net(lrhs_size=(32, 32, 3), hrms_size = (32, 32, 1)):
