#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/7/7
import numpy as np
import scipy.io as sio
import math
from skimage.metrics import structural_similarity as ssimindex


def quality (imagery1, imagery2):
    imagery1 = imagery1 
    imagery2 = imagery2
    Nway = imagery1.shape
    psnr = np.zeros((Nway[2], 1))
    ssim = np.zeros((Nway[2], 1))
    for i in range(Nway[2]):
        psnr[i] = psnr_index(imagery1[:,:,i]*255,imagery2[:,:,i]*255)
        ssim[i] = ssimindex(imagery1[:,:,i],imagery2[:,:,i], data_range=1)
    psnr = np.mean(psnr)
    ssim = np.mean(ssim)
    return psnr, ssim

def psnr_index (x, y):
    mse = np.mean((x-y)**2)
    p = 10 * math.log10(255.0**2/mse)
    return p
