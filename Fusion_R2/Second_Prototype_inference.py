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
import imageio
import torchvision
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import h5py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import random


# Set random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

hf = h5py.File(r"C:\Users\ishan\Desktop\Image Fusion\Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')	
val_data = hf['data'][()]
hf.close()
for i in range(len(val_data)):
    for j in range(4):
        val_data[i,j,:,:] =(val_data[i,j,:,:] - np.min(val_data[i,j,:,:])) / (np.max(val_data[i,j,:,:]) - np.min(val_data[i,j,:,:]))
        
val_t1ce_tensor = torch.from_numpy(val_data[:,2,:,:]).float()