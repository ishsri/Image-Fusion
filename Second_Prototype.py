import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import glob
import natsort
import numpy as np
import pandas as pd
import imageio
import torchvision
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import h5py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import random

hf = h5py.File(r"/home/h3/issr292b/image_fusion/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
train_data = hf['data'][()]     #`data` is now an ndarray
train_data_tensor = torch.from_numpy(train_data).float()

hf.close()

for i in range(len(train_data)):
    for j in range(4):
        train_data[i,j,:,:] = (train_data[i,j,:,:] - np.min(train_data[i,j,:,:])) / (np.max(train_data[i,j,:,:]) - np.min(train_data[i,j,:,:]))


train_data_t1ce = train_data[:,2,:,:]
train_data_flair = train_data[:,3,:,:]
height = train_data.shape[2]
width = train_data.shape[3]
total_train_images = train_data.shape[0]
r_channel = np.zeros((total_train_images, height, width, 3))
r_channel[:,:,:,0] = train_data_t1ce
r_channel[:,:,:,1] = np.zeros((total_train_images, height, width))
r_channel[:,:,:,2] = np.zeros((total_train_images, height, width))

g_channel = np.zeros((total_train_images, height, width, 3))
g_channel[:,:,:,1] = train_data_flair
g_channel[:,:,:,0] = np.zeros((total_train_images, height, width))
g_channel[:,:,:,2] = np.zeros((total_train_images, height, width))

r_channel_transpose = np.transpose(r_channel, [0,2,1,3])
r_channel_transpose = np.transpose(r_channel, [0,3,1,2])

g_channel_transpose = np.transpose(g_channel, [0,2,1,3])
g_channel_transpose = np.transpose(g_channel, [0,3,1,2])

hf_new = h5py.File('/home/h3/issr292b/image_fusion/fusion_r2_validation_data.h5', 'w')
hf_new.create_dataset('g_channel', data = g_channel_transpose)
hf_new.create_dataset('r_channel', data = r_channel_transpose)
hf_new.close()
