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

hf = h5py.File(r"/projects/p084/p_discoret/fusion_r2_validation_data.h5", 'r')
val_flair_tensor = hf['g_channel'][()]
val_t1ce_tensor = hf['r_channel'][()]
hf.close()
val_data_tensor = np.concatenate((val_t1ce_tensor, val_flair_tensor), axis=1)



#define the network

class def_model(nn.Module):
    def  __init__(self):
        super(def_model, self).__init__()
        
        ###############
        #Encoder
        ###############
        self.conv1 = nn.Sequential( #input shape (,2,240,240)
                         nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,32,240,240)   
        ##### res like layer 1#####
        self.res1 = nn.Sequential(  #input shape (,32,240,240)
                         nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size  = 3, stride= 1, padding = 1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels  = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                         nn.ReLU()) #output shape (,32,240,240)
        ##### downsample conv like layer#####
        self.conv2 = nn.Sequential(  #input shape (,32,240,240)
                         nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size  = 4, stride= 2, padding = 1),
                         nn.ReLU()) #output shape (,64,120,120)        
        #####res like layer 2#####
        self.res2 = nn.Sequential( #input shape (,64,120,120)
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,64,120,120) 
        #####conv like layer#####
        self.conv3 = nn.Sequential(  #input shape (,64,120,120)
                         nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size  = 3, stride= 1, padding = 1),
                         nn.ReLU()) #output shape (,64,120,120)
        #####res like layer 3#####
        self.res3  =     nn.Sequential( #input shape (,64,120,120)
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,64,120,120) 
        ##### downsample conv like layer 2#####
        self.conv4 = nn.Sequential(  #input shape (,64,120,120)
                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size  = 4, stride= 2, padding = 1),
                         nn.ReLU()) #output shape (,128,60,60) 
        #####res like layer 4#####
        self.res4 = nn.Sequential( #input shape (,128,60,60)
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,128,60,60) 
        #####conv like layer#####
        self.conv5 = nn.Sequential(  #input shape (,128,60,60)
                         nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size  = 3, stride= 1, padding = 1),
                         nn.ReLU()) #output shape (,128,60,60)
        #####res like layer 5#####
        self.res5  =     nn.Sequential( #input shape (,128,60,60)
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,128,60,60) 
        ##### downsample conv like layer 3#####
        self.conv6 = nn.Sequential(  #input shape (,128,60,60)
                         nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size  = 4, stride= 2, padding = 1),
                         nn.ReLU()) #output shape (,256,30,30) 
        #####res like layer 6#####
        self.res6 = nn.Sequential( #input shape (,256,30,30)
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,256,30,30) 
        #####conv like layer#####
        self.conv7 = nn.Sequential(  #input shape (,256,30,30)
                         nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size  = 3, stride= 1, padding = 1),
                         nn.ReLU()) #output shape (,256,30,30)
        #####res like layer 7#####
        self.res7  =     nn.Sequential( #input shape (,256,30,30)
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,256,30,30) 
        #####conv like layer#####
        self.conv8 = nn.Sequential(  #input shape (,256,30,30)
                         nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size  = 3, stride= 1, padding = 1),
                         nn.ReLU()) #output shape (,256,30,30) 
        #####res like layer 8#####
        self.res8 = nn.Sequential( #input shape (,256,30,30)
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,256,30,30) 
        #####conv like layer#####
        self.conv9 = nn.Sequential(  #input shape (,256,30,30)
                         nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size  = 3, stride= 1, padding = 1),
                         nn.ReLU()) #output shape (,256,30,30)
        #####res like layer 9#####
        self.res9  =     nn.Sequential( #input shape (,256,30,30)
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                         nn.ReLU()) #output shape (,256,30,30)
        ###############
        #Decoder
        ###############
        ##### upsample conv like layer 3#####
        self.conv10 = nn.Sequential(  #input shape (,256,30,30)
                         nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size  = 3, stride= 1, padding = 1),
                         nn.Upsample(scale_factor=2, mode='nearest')) #output shape (,128,60,60)
        
        #####res like layer 10#####
        self.res10  =    nn.Sequential( #input shape (,128,60,60)
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                         #nn.ReLU(),
                         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)#,
        )#nn.ReLU()) #output shape (,128,60,60)
        
        ##### upsample conv like layer 4#####
        self.conv11 = nn.Sequential(  #input shape (,128,60,60)
                         nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size  = 3, stride= 1, padding = 1),
                         nn.Upsample(scale_factor=2, mode='nearest')) #output shape (,64,120,120)
        
        #####res like layer 11#####
        self.res11  =    nn.Sequential( #input shape (,64,120,120)
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                         #nn.ReLU(),
                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)#,
        )#nn.ReLU()) #output shape (,64,120,120)   
        
        ##### upsample conv like layer 4#####
        self.conv12 = nn.Sequential(  #input shape (,64,120,120)
                         nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size  = 3, stride= 1, padding = 1),
                         nn.Upsample(scale_factor=2, mode='nearest')) #output shape (,32,240,240)
        
        #####res like layer 11#####
        self.res12  =    nn.Sequential( #input shape (,32,240,240)
                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                         #nn.ReLU(),
                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)#,
        )#nn.ReLU()) #output shape (,32,240,240)       
        
        #####conv like layer#####
        self.conv13 = nn.Sequential(  #input shape (,32,240,240)
                         nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size  = 3, stride= 1, padding = 1)) #output shape (,1,240,240)
 
        #####sigmoid layer#####
        self.sigmoid1 = torch.nn.Sigmoid()
        
    def forward(self, x):
        #conv1
        x1 = self.conv1(x)
        #res1
        x2 = self.res1(x1)
        #conv2 
        x3 = self.conv2(x2)
        #res2
        x4 = self.res2(x3)
        #conv3
        x5 = self.conv3(x4)
        #res3
        x6 = self.res3(x5)
        #conv4
        x7 = self.conv4(x6)
        #res4
        x8 = self.res4(x7)
        #conv5
        x9 = self.conv5(x8)  
        #res5
        x10 = self.res5(x9)
        #conv6
        x11 = self.conv6(x10)
        #res6
        x12 = self.res6(x11)
        #conv7
        x13 = self.conv7(x12)
        #res7
        x14 = self.res7(x13)
        #conv8
        x15 = self.conv8(x14)
        #res8
        x16 = self.res8(x15)
        #conv9
        x17 = self.conv9(x16)
        #res9
        x18 = self.res9(x17)
        #conv10
        x19 = self.conv10(x18)
        # add operation
        add1 = x19 + x10
        #res10
        x20 = self.res10(add1)
        #conv11
        x21 = self.conv11(x20)
        #add operation
        add2 = x21 + x6
        #res11
        x22 = self.res11(add2)
        #conv12
        x23 = self.conv12(x22)
        #add operation
        add3 = x23 + x2
        #res12
        x24 = self.res12(add3)
        #conv13
        x25 = self.conv13(x24)
        #sigmoid 
        x26 = self.sigmoid1(x25)
        return x26
        #execute the network

        
model = def_model().to(device)
gpu_ids = [0,1]
model = model.float()
if device == 'cuda':
    net = torch.nn.DataParallel(model, gpu_ids)
    cudnn.benchmark = True
        
mod = torch.load(r"/home/h3/issr292b/image_fusion/epoch/model_r2_0.49.pt")
model_state_dict = mod["model_state_dict"]
model.load_state_dict(model_state_dict)
model.eval()


val_data_tensor = torch.from_numpy(val_data_tensor).float()
total_val_images = 1153

for i in range(0, total_val_images):
    with torch.no_grad():
        input_val  = val_data_tensor[i:i+1,:,:,:].to(device)
        #input_val = torch.unsqueeze(input_val, 0)
        weight_map_val = model(input_val)
        torchvision.utils.save_image(weight_map_val, '/home/h3/issr292b/image_fusion/inference_r2/0.49/weight_map/count_{}.png'.format(i))