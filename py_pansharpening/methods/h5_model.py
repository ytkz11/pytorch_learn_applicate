#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/5/29 13:36 
# @Author : DKY
# @File : h5_model.py

from keras.models import load_model
from keras import backend as K
from keras.layers import Concatenate, Conv2D, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizer_v2.adam  import Adam
from keras.models import Model
from osgeo import gdal
import numpy as np
from utils import upsample_interp23, downgrade_images
import random
from tqdm import tqdm
import cv2
from scipy import signal
def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true*255 - y_pred*255), axis=(-3, -2, -1))
    return K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10))
def pnn_net(lrhs_size=(32, 32, 3), hrms_size=(32, 32, 1)):
    lrhs_inputs = Input(lrhs_size)
    hrms_inputs = Input(hrms_size)

    mixed = Concatenate()([lrhs_inputs, hrms_inputs])

    mixed1 = Conv2D(64, (9, 9), strides=(1, 1), padding='same', activation='relu')(mixed)

    mixed1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(mixed1)

    c6 = Conv2D(lrhs_size[2], (5, 5), strides=(1, 1), padding='same', activation='relu', name='model1_last1')(mixed1)

    model = Model(inputs=[lrhs_inputs, hrms_inputs], outputs=c6)

    model.compile(optimizer=Adam(lr=5e-4), loss='mse', metrics=[psnr])

    model.summary()

    return model
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
    mss = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/mss1.tif'
    pan = 'D:\dengkaiyuan\code\pytorch_learn_applicate\py_pansharpening\images/pan1.tif'

    original_msi = read_tif_to_np(mss)
    original_pan = read_tif_to_np(pan)

    '''normalization'''
    max_patch, min_patch = np.max(original_msi, axis=(0, 1)), np.min(original_msi, axis=(0, 1))
    original_msi = np.float32(original_msi - min_patch) / (max_patch - min_patch)

    max_patch, min_patch = np.max(original_pan, axis=(0, 1)), np.min(original_pan, axis=(0, 1))
    original_pan = np.float32(original_pan - min_patch) / (max_patch - min_patch)

    '''generating ms image with gaussian kernel'''
    sig = (1 / (2 * (2.772587) / 4 ** 2)) ** 0.5
    kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9, sig).T)
    new_lrhs = []
    for i in range(original_msi.shape[-1]):
        temp = signal.convolve2d(original_msi[:, :, i], kernel, boundary='wrap', mode='same')
        temp = np.expand_dims(temp, -1)
        new_lrhs.append(temp)
    new_lrhs = np.concatenate(new_lrhs, axis=-1)
    used_ms = new_lrhs[0::4, 0::4, :]

    # '''generating ms image with bicubic interpolation'''
    # used_ms = cv2.resize(original_msi, (original_msi.shape[1]//4, original_msi.shape[0]//4), cv2.INTER_CUBIC)

    '''generating pan image with gaussian kernel'''
    used_pan = signal.convolve2d(original_pan, kernel, boundary='wrap', mode='same')
    used_pan = np.expand_dims(used_pan, -1)
    used_pan = used_pan[0::4, 0::4, :]

    used_ms = original_msi
    used_pan = original_pan
    used_pan = np.expand_dims(used_pan, -1)
    stride = 8
    training_size = 32  # training patch size
    testing_size = 400  # testing patch size
    reconstructing_size = 320  # reconstructing patch size
    left_pad = (testing_size - reconstructing_size) // 2
    lrhs = used_ms
    hrms= used_pan
    M, N, c = hrms.shape
    m, n, C = lrhs.shape

    model = pnn_net(lrhs_size=(testing_size, testing_size, C), hrms_size=(testing_size, testing_size, c))
    model = load_model('my_model.h5',custom_objects = {"psnr": psnr })
    model.summary()

    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M/m)) == int(np.round(N/n))
    train_hrhs_all = []
    train_hrms_all = []
    train_lrhs_all = []

    used_hrhs = lrhs
    used_lrhs = lrhs
    used_lrhs, used_hrms = downgrade_images(used_lrhs, hrms, ratio, sensor=None)
    print(used_lrhs.shape, used_hrms.shape)
    used_lrhs = upsample_interp23(used_lrhs, ratio)
    """crop images"""
    print('croping images...')

    for j in range(0, used_hrms.shape[0] - training_size, stride):
        for k in range(0, used_hrms.shape[1] - training_size, stride):
            temp_hrhs = used_hrhs[j:j + training_size, k:k + training_size, :]
            temp_hrms = used_hrms[j:j + training_size, k:k + training_size, :]
            temp_lrhs = used_lrhs[j:j + training_size, k:k + training_size, :]

            train_hrhs_all.append(temp_hrhs)
            train_hrms_all.append(temp_hrms)
            train_lrhs_all.append(temp_lrhs)

    train_hrhs_all = np.array(train_hrhs_all, dtype='float16')
    train_hrms_all = np.array(train_hrms_all, dtype='float16')
    train_lrhs_all = np.array(train_lrhs_all, dtype='float16')

    index = [i for i in range(train_hrhs_all.shape[0])]
    #    random.seed(2020)
    random.shuffle(index)
    train_hrhs = train_hrhs_all[index, :, :, :]
    train_hrms = train_hrms_all[index, :, :, :]
    train_lrhs = train_lrhs_all[index, :, :, :]

    print(train_hrhs.shape, train_hrms.shape, train_lrhs.shape)
    new_M = min(M, m * ratio)
    new_N = min(N, n * ratio)

    print('output image size:', new_M, new_N)

    test_label = np.zeros((new_M, new_N, C), dtype='uint8')

    used_lrhs = lrhs[:new_M // ratio, :new_N // ratio, :]
    used_hrms = hrms[:new_M, :new_N, :]

    used_lrhs = upsample_interp23(used_lrhs, ratio)

    used_lrhs = np.expand_dims(used_lrhs, 0)
    used_hrms = np.expand_dims(used_hrms, 0)

    used_lrhs = np.pad(used_lrhs, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)),
                       mode='symmetric')
    used_hrms = np.pad(used_hrms, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)),
                       mode='symmetric')

    for h in tqdm(range(0, new_M, reconstructing_size)):
        for w in range(0, new_N, reconstructing_size):
            temp_lrhs = used_lrhs[:, h:h + testing_size, w:w + testing_size, :]
            temp_hrms = used_hrms[:, h:h + testing_size, w:w + testing_size, :]

            fake = model.predict([temp_lrhs, temp_hrms])
            fake = np.clip(fake, 0, 1)
            fake.shape = (testing_size, testing_size, C)
            fake = fake[left_pad:(testing_size - left_pad), left_pad:(testing_size - left_pad)]
            fake = np.uint8(fake * 255)

            if h + testing_size > new_M:
                fake = fake[:new_M - h, :, :]

            if w + testing_size > new_N:
                fake = fake[:, :new_N - w, :]

            test_label[h:h + reconstructing_size, w:w + reconstructing_size] = fake
        fused_image = np.uint8(test_label)
    save_channels = [0, 1, 2]  # BGR-NIR for GF2
    cv2.imwrite('PNN2.tiff', fused_image[:, :, save_channels])
    a = 0