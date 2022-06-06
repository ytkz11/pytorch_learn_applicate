#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/6/6 8:53 
# @Author : DKY
# @File : use_pnn.py
from keras.models import load_model
from keras import backend as K
from keras.layers import Concatenate, Conv2D, Input
from keras.optimizer_v2.adam import Adam
from keras.models import Model
from osgeo import gdal, gdalconst
import numpy as np
from utils import upsample_interp23, downgrade_images
import random
from tqdm import tqdm
import cv2
from skimage.transform import resize
from hh_hv_fusion import read_tif_to_np
import os, math, time
import warnings as warn
from hh_hv_fusion import nearest


def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true * 255 - y_pred * 255), axis=(-3, -2, -1))
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


def check_resampled_multispectralfile_exsit(self):
    sourceDataset = gdal.Open(self.mssfile, gdal.GA_ReadOnly)
    dstDataset = gdal.Open(self.panfile, gdal.GA_ReadOnly)
    bandnum = sourceDataset.RasterCount
    if self.mssfile.endswith('.tif'):
        resampledMultispectralGeotiffFilename = self.mssfile.replace('.tif', '_RESAMPLED.tiff')
    elif self.mssfile.endswith('.tiff'):
        resampledMultispectralGeotiffFilename = self.mssfile.replace('.tiff', '_RESAMPLED.tiff')
    elif self.mssfile.endswith('.dat'):
        resampledMultispectralGeotiffFilename = self.mssfile.replace('.dat', '_RESAMPLED.tiff')
    else:
        print(
            '  \n    Multispectral Geotiff image file: ' + self.mssfile + ' should have .TIF or .tif extension. Exiting ... ')
        raise Exception[
            'Multispectral Geotiff image file: ' + self.mssfile + ' should have .TIF or .tif extension. Exiting ...']
    file_exsit = os.path.isfile(resampledMultispectralGeotiffFilename)

    if file_exsit == True:
        print('resampled的多光谱文件存在，检查其是否完整')
        pan_size = float(os.path.getsize(self.panfile) / (1024.0 * 1024.0 * 1024.0))
        resampled_size = float(os.path.getsize(resampledMultispectralGeotiffFilename) / (1024.0 * 1024.0 * 1024.0))
        print('bandnum x pan_size: %f GB' % (pan_size * bandnum))
        print('resampled_size: %f GB' % resampled_size)
        if resampled_size < pan_size * bandnum + 0.5 and resampled_size > pan_size * bandnum - 0.5:

            state = False
        else:
            state = True

        if state == True:
            print('resampled的多光谱文件不完整，删除它，准备重新生成')

            os.remove(resampledMultispectralGeotiffFilename)
            self.resample(sourceDataset, dstDataset, resampledMultispectralGeotiffFilename,
                          gdalconst.GRA_Bilinear)
        else:
            print('resampled的多光谱文件完整，不需重新生成')
    else:
        print('resampled的多光谱文件不存在，需重新生成')
        self.resample(sourceDataset, dstDataset, resampledMultispectralGeotiffFilename,
                      gdalconst.GRA_Bilinear)
    return resampledMultispectralGeotiffFilename


class PNN_tif():
    def __init__(self, mssfile, panfile, out=None):
        self.mssfile = mssfile
        self.panfile = panfile
        if out == None:
            self.out = os.path.dirname(self.mssfile)
        else:
            self.out = out
        a = gdal.Open(self.mssfile)
        if a.RasterCount < 3:
            print('多光谱的波段少于三波段')
            raise Exception('多光谱的波段少于三波段!')
        del a

    def pansharpenPNN(self, resampledMultispectralGeotiffFilename, outname):
        resampledMultispectralGeotiffFilename = self.check_resampled_multispectralfile_exsit()
        outnameBrovey = resampledMultispectralGeotiffFilename.replace(
            '_RESAMPLED.tiff', '_panSharpenedBrovey.tiff')
        outnameBrovey = os.path.join(self.out, os.path.basename(outnameBrovey))

        with warn.catch_warnings():
            warn.filterwarnings('ignore', category=RuntimeWarning)

        dsPan = gdal.Open(self.panfile)
        dsMulti = gdal.Open(resampledMultispectralGeotiffFilename)

        # create output directory to hold .dat files (binary files)
        outputDir = os.path.dirname(self.panfile)
        nrows, ncols = dsPan.RasterYSize, dsPan.RasterXSize
        band_num = dsMulti.RasterCount

        if os.path.isfile(outname):
            os.remove(outname)

        Driver = dsMulti.GetDriver()
        geoTransform1 = dsMulti.GetGeoTransform()
        proj1 = dsMulti.GetProjection()
        ListgeoTransform1 = list(geoTransform1)
        # write the multispectral geotiff

        dst = Driver.Create(outname, ncols, nrows, band_num, gdal.GDT_Float32)
        dst.SetGeoTransform(geoTransform1)
        # dst.SetProjection(dsMulti.GetProjection())
        dst.SetProjection(proj1)

        nBlockSize = 1024  # * 2 * 2  # 块大小为1024*1024
        i = 0
        j = 0
        # 进度条参数
        XBlockcount = math.ceil(ncols / nBlockSize)
        YBlockcount = math.ceil(nrows / nBlockSize)

        try:
            with tqdm(total=XBlockcount * YBlockcount, iterable='iterable', desc='Brovey') as pbar:
                while i < nrows:
                    while j < ncols:
                        # 保存分块大小
                        nXBK = nBlockSize
                        nYBK = nBlockSize

                        # 最后不够分块的区域，有多少读取多少
                        if i + nBlockSize > nrows:
                            nYBK = nrows - i
                        if j + nBlockSize > ncols:
                            nXBK = ncols - j

                        # 建立一个nBlockSize x nBlockSize x （波段数+1）的空矩阵，
                        # 波段数+1：多光谱波段 + 一个全色波段
                        img_arr = np.zeros(shape=(nYBK, nXBK))
                        # 分块读取影像
                        pandata = dsPan.GetRasterBand(1)
                        img_arr[:, :] = pandata.ReadAsArray(j, i, nXBK, nYBK).astype(float)

                        M, N = img_arr.shape

                        img_mss = np.zeros(shape=(int(nYBK / 4), int(nXBK / 4), band_num))

                        for n in range(band_num):
                            banadata = dsMulti.GetRasterBand(n + 1)
                            banadata.SetNoDataValue(-9999)
                            original_msi = banadata.ReadAsArray(j, i, nXBK, nYBK).astype(float)
                            used_ms = nearest(original_msi, (int(M / 4), int(N / 4)))
                            img_mss[:, :, n] = used_ms[:,:,0]

                        fused_image = self.use_pnn(img_mss, img_arr)

                        x, y, C = fused_image.shape
                        for m in range(1, C + 1):
                            outband = dst.GetRasterBand(m)
                            outband.SetNoDataValue(-9999)

                            outband.WriteArray(fused_image[:, :, m - 1], j, i)

                        j = j + nXBK
                        time.sleep(1)
                        pbar.update(1)
                    j = 0
                    i = i + nYBK
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

    def check_resampled_multispectralfile_exsit(self):
        sourceDataset = gdal.Open(self.mssfile, gdal.GA_ReadOnly)
        dstDataset = gdal.Open(self.panfile, gdal.GA_ReadOnly)
        bandnum = sourceDataset.RasterCount
        if self.mssfile.endswith('.tif'):
            resampledMultispectralGeotiffFilename = self.mssfile.replace('.tif', '_RESAMPLED.tiff')
        elif self.mssfile.endswith('.tiff'):
            resampledMultispectralGeotiffFilename = self.mssfile.replace('.tiff', '_RESAMPLED.tiff')
        elif self.mssfile.endswith('.dat'):
            resampledMultispectralGeotiffFilename = self.mssfile.replace('.dat', '_RESAMPLED.tiff')
        else:
            print(
                '  \n    Multispectral Geotiff image file: ' + self.mssfile + ' should have .TIF or .tif extension. Exiting ... ')
            raise Exception[
                'Multispectral Geotiff image file: ' + self.mssfile + ' should have .TIF or .tif extension. Exiting ...']
        file_exsit = os.path.isfile(resampledMultispectralGeotiffFilename)

        if file_exsit == True:
            print('resampled的多光谱文件存在，检查其是否完整')
            pan_size = float(os.path.getsize(self.panfile) / (1024.0 * 1024.0 * 1024.0))
            resampled_size = float(os.path.getsize(resampledMultispectralGeotiffFilename) / (1024.0 * 1024.0 * 1024.0))
            print('bandnum x pan_size: %f GB' % (pan_size * bandnum))
            print('resampled_size: %f GB' % resampled_size)
            if resampled_size < pan_size * bandnum + 0.5 and resampled_size > pan_size * bandnum - 0.5:

                state = False
            else:
                state = True

            if state == True:
                print('resampled的多光谱文件不完整，删除它，准备重新生成')

                os.remove(resampledMultispectralGeotiffFilename)
                self.resample(sourceDataset, dstDataset, resampledMultispectralGeotiffFilename,
                              gdalconst.GRA_Bilinear)
            else:
                print('resampled的多光谱文件完整，不需重新生成')
        else:
            print('resampled的多光谱文件不存在，需重新生成')
            self.resample(sourceDataset, dstDataset, resampledMultispectralGeotiffFilename,
                          gdalconst.GRA_Bilinear)
        return resampledMultispectralGeotiffFilename

    def runPNN(self):
        resampledMultispectralGeotiffFilename = self.check_resampled_multispectralfile_exsit()
        outnamePNN = resampledMultispectralGeotiffFilename.replace(
            '_RESAMPLED.tiff', '_panSharpenedPNN.tiff')
        outnamePNN = os.path.join(self.out, os.path.basename(outnamePNN))
        self.pansharpenPNN(resampledMultispectralGeotiffFilename, outnamePNN)
        os.remove(resampledMultispectralGeotiffFilename)

    @staticmethod
    def resample(sourceDataset, dstDataset, outname, interp):
        '''
        参数:
        srcImageFilename (str):源(低分辨率)多光谱Geotiff文件名。
        sourceDataset (osgeo.gdal.Dataset):输入多光谱GDAL dataset对象。
        dstDataset (osgeo.gdal.Dataset):目标(高分辨率)全色数据集对象。
        outname (str):重新采样后输出的Geotiff的名称
        interp (int): GDAL插值方法(即gdalconstt . gra_cubic)
        '''
        print('resample image')
        # get the "source" (i.e. low-res. multispectral) projection and geotransform
        srcProjection = sourceDataset.GetProjection()
        srcGeotransform = sourceDataset.GetGeoTransform()
        srcNumRasters = sourceDataset.RasterCount
        dstProjection = dstDataset.GetProjection()
        dstGeotransform = dstDataset.GetGeoTransform()
        nrows = dstDataset.RasterYSize
        ncols = dstDataset.RasterXSize
        dst_fn = outname

        # if the resampled-multispectral (3 or 4 band) Geotiff image file exists, delete it.
        if not os.path.isfile(outname):
            dst_ds = gdal.GetDriverByName('GTiff').Create(dst_fn, ncols, nrows, srcNumRasters, gdalconst.GDT_Float32)
            dst_ds.SetGeoTransform(dstGeotransform)
            dst_ds.SetProjection(dstProjection)
            gdal.ReprojectImage(sourceDataset, dst_ds, srcProjection, dstProjection, interp)
            dst_ds = None
            del dst_ds
        print('完成resample')
        return dst_fn

    @staticmethod
    def use_pnn(mss, pan):
        original_msi = mss
        original_pan = pan

        '''normalization'''
        max_patch, min_patch = np.max(original_msi, axis=(0, 1)), np.min(original_msi, axis=(0, 1))
        original_msi = np.float32(original_msi - min_patch) / (max_patch - min_patch)

        max_patch, min_patch = np.max(original_pan, axis=(0, 1)), np.min(original_pan, axis=(0, 1))
        original_pan = np.float32(original_pan - min_patch) / (max_patch - min_patch)

        '''generating ms image with gaussian kernel'''
        sig = (1 / (2 * (2.772587) / 4 ** 2)) ** 0.5
        kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9, sig).T)
        new_lrhs = []

        # gf3
        used_pan = original_pan
        used_pan = np.expand_dims(used_pan, -1)
        used_ms = original_msi
        M, N, c = used_pan.shape

        stride = 8
        training_size = 32  # training patch size
        testing_size = 400  # testing patch size
        reconstructing_size = 320  # reconstructing patch size
        left_pad = (testing_size - reconstructing_size) // 2

        M, N, c = used_pan.shape
        m, n, C = used_ms.shape

        model = pnn_net(lrhs_size=(testing_size, testing_size, C), hrms_size=(testing_size, testing_size, c))
        model = load_model('my_model.h5', custom_objects={"psnr": psnr})
        model.summary()

        ratio = int(np.round(M / m))
        print('get sharpening ratio: ', ratio)
        if int(np.round(M / m)) == int(np.round(N / n)):
            used_ms = original_msi
        else:
            used_ms = resize(original_msi, (int(M / 4), int(N / 4)))
        lrhs = used_ms
        hrms = used_pan
        assert int(np.round(M / m)) == int(np.round(N / n))
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
        try:
            train_hrhs = train_hrhs_all[index, :, :, :]
            train_hrms = train_hrms_all[index, :, :, :]
            train_lrhs = train_lrhs_all[index, :, :, :]
        except:
            print('e')
            return np.zeros(shape=(M, N,C))
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
        cv2.imwrite('PNN6.tiff', fused_image[:, :, save_channels])
        a = 0
        return fused_image


if __name__ == '__main__':
    mssfile = r'D:\dengkaiyuan\code\code1029\GF\fusion\pms.tif'
    panfile = r'D:\dengkaiyuan\code\code1029\GF\fusion\pan.tif'
    b = PNN_tif(mssfile, panfile)
    b.runPNN()
    type = 0
