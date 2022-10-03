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


#hf = h5py.File("/home/h3/issr292b/image_fusion/fusion_r2_training_data.h5", 'r')
#train_data = hf['data'][()]     #`data` is now an ndarray
#hf.close()

#hf = h5py.File("/projects/p084/p_discoret/Brats2018_training_data_sep_channels_train_val_mix.h5", 'r')
#train_labels = hf['label'][()]     #`data` is now an ndarray
#hf.close()

#hf = h5py.File("/projects/p084/p_discoret/fusion_r2_validation_data.h5", 'r')
#val_data = hf['data'][()]     #`data` is now an ndarray
#hf.close()

#hf = h5py.File("/projects/p084/p_discoret/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
#val_labels = hf['label'][()]     #`data` is now an ndarray
#hf.close()



#for i in range(len(train_data)):
    #for j in range(4):
        #train_data[i,j,:,:] =(train_data[i,j,:,:] - np.min(train_data[i,j,:,:])) / (np.max(train_data[i,j,:,:]) - np.min(train_data[i,j,:,:]))
        
        
#for i in range(len(val_data)):
    #for j in range(4):
        #val_data[i,j,:,:] =(val_data[i,j,:,:] - np.min(val_data[i,j,:,:])) / (np.max(val_data[i,j,:,:]) - np.min(val_data[i,j,:,:]))
		
hf = h5py.File("/projects/p084/p_discoret/fusion_r2_training_data.h5", 'r')	
train_flair_tensor = hf['g_channel'][()]
train_t1ce_tensor = hf['r_channel'][()]
hf.close()
train_data_tensor = np.concatenate((train_t1ce_tensor, train_flair_tensor), axis=1)
train_data_tensor = torch.from_numpy(train_data_tensor).float()
#np.delete(train_flair_tensor[:])
#del train_t1ce_tensor[:]
train_flair_tensor = []
train_t1ce_tensor = []

hf = h5py.File("/projects/p084/p_discoret/fusion_r2_validation_data.h5", 'r')
val_flair_tensor = hf['g_channel'][()]
val_t1ce_tensor = hf['r_channel'][()]
hf.close()
val_data_tensor = np.concatenate((val_t1ce_tensor, val_flair_tensor), axis=1)
val_data_tensor = torch.from_numpy(val_data_tensor).float()
#del val_flair_tensor[:]
#del val_t1ce_tensor[:]
val_flair_tensor = []
val_t1ce_tensor = []

#do it for the training data
#train_t1_tensor = torch.from_numpy(train_data[:,0,:,:]).float()
# train_t1_tensor = train_t1_tensor[:, None, :, :]

# train_t2_tensor = torch.from_numpy(train_data[:,1,:,:]).float()
# train_t2_tensor = train_t2_tensor[:, None, :, :]

# train_t1ce_tensor = torch.from_numpy(train_data[:,2,:,:]).float()
# train_t1ce_tensor = train_t1_tensor[:, None, :, :]

# train_flair_tensor = torch.from_numpy(train_data[:,3,:,:]).float()
# train_flair_tensor = train_flair_tensor[:, None, :, :]

# train_label_tensor = torch.from_numpy(train_labels).float()


# #do it for the validation data
# val_t1_tensor = torch.from_numpy(val_data[:,0,:,:]).float()
# val_t1_tensor = val_t1_tensor[:, None, :, :]

# val_t2_tensor = torch.from_numpy(val_data[:,1,:,:]).float()
# val_t2_tensor = val_t2_tensor[:, None, :, :]

# val_t1ce_tensor = torch.from_numpy(val_data[:,2,:,:]).float()
# val_t1ce_tensor = val_t1ce_tensor[:, None, :, :]

# val_flair_tensor = torch.from_numpy(val_data[:,3,:,:]).float()
# val_flair_tensor = val_flair_tensor[:, None, :, :]

# val_label_tensor = torch.from_numpy(val_labels).float()


# #do it for the training data
# train_data_tensor = torch.from_numpy(train_data).float()
# train_label_tensor = torch.from_numpy(train_labels).float()

# val_data_tensor = torch.from_numpy(val_data).float()
# val_label_tensor = torch.from_numpy(val_labels).float()



total_train_images = 8500
total_val_images = 1153
EPOCHS = 101
batch_size = 64
gpu_ids = [0,1]




#define the network
class def_model(nn.Module):
    def  __init__(self):
        super(def_model, self).__init__()
        # Define the model based on the paper https://arxiv.org/abs/1810.11654
        ###############
        #Encoder
        ###############
        self.conv1 = nn.Sequential( #input shape (,6,240,240)
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
model = model.float()
if device == 'cuda':
    net = torch.nn.DataParallel(model, gpu_ids)
    cudnn.benchmark = True
        
#define the optimizers and loss functions 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)   # optimize all cnn parameters
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 0.95 ** epoch)

l2_loss   = nn.MSELoss() #MSEloss  
    
lamda_ssim = 0.1
lamda_l2 = 0.49
lamda_fusion = 0.99
    
loss_train = []
ssim_t1ce_train = []
ssim_flair_train = []
ssim_t1ce_val = []
ssim_flair_val = []
loss_ssim_train_t1ce = []
loss_ssim_train_flair = []
loss_l2_train_t1ce = []
loss_l2_train_flair = []
        
loss_val = []
loss_ssim_val_t1ce = []
loss_ssim_val_flair = []
loss_l2_val_t1ce = []
loss_l2_val_flair = []
    
ep_ssim_t1ce_train = []
ep_ssim_flair_train = []    
ep_ssim_t1ce_val = []
ep_ssim_flair_val = []
ep_train_loss = []
ep_ssim_train_loss_t1ce = []
ep_ssim_train_loss_flair = []
ep_l2_train_loss_t1ce = []
ep_l2_train_loss_flair = []

ep_val_loss = []
ep_ssim_val_loss_t1ce = []
ep_ssim_val_loss_flair = []
ep_l2_val_loss_t1ce = []
ep_l2_val_loss_flair = []
count = 0         
#train the model
for epoch in range(EPOCHS):    
    model.train()    
    # run batch images
    batch_idxs = total_train_images // batch_size
    for idx in range(0, batch_idxs):
        input_  = train_data_tensor[idx*batch_size : (idx+1)*batch_size,:,:,:].to(device)
        weight_map = model(input_)
        fused = weight_map*input_[:,0:1,:,:] + (1-weight_map)*input_[:,1:2,:,:]

        
        path_im = f"/home/h3/issr292b/image_fusion/train_samples/en-de/ssim_0.1/"
        path_fused = f"/home/h3/issr292b/image_fusion/train_samples_fused/en-de/ssim_0.1/"
        os.makedirs(path_im, exist_ok=True)
        os.makedirs(path_fused, exist_ok=True)
        images_concat = torchvision.utils.make_grid(weight_map, nrow=int(weight_map.shape[0] ** 0.5), padding=2, pad_value=255)
        fused_save = torchvision.utils.make_grid(fused, nrow = int(fused.shape[0] ** 0.5), padding=2, pad_value=255)
        if count % 10 == 0:
            torchvision.utils.save_image(images_concat, '/home/h3/issr292b/image_fusion/train_samples/en-de/ssim_0.1/count_{}.png'.format(count))
            torchvision.utils.save_image(fused_save, '/home/h3/issr292b/image_fusion/train_samples_fused/en-de/ssim_0.1/count_{}.png'.format(count))
          
        #SSIM loss for the fusion training
        ssim_module = SSIM(data_range=255, size_average=True, channel=3)
        ssim_loss_t1ce  = 1 - ssim_module(weight_map, input_[:,0:3,:,:])
        ssim_loss_flair = 1 - ssim_module(weight_map, input_[:,3:6,:,:])
        #club the T1ce and flair ssim losses
        ssim_loss_combined = lamda_ssim * ssim_loss_t1ce + (1-lamda_ssim) * ssim_loss_flair
        
        #l2 loss for the fusion training
        l2_loss_t1ce  = l2_loss(weight_map, input_[:,0:3,:,:])
        l2_loss_flair = l2_loss(weight_map, input_[:,3:6,:,:])        
        #club the T1ce and flair l2 losses
        l2_loss_combined = lamda_l2 * l2_loss_t1ce + (1-lamda_l2) * l2_loss_flair        
        
        #combine ssim_l2_loss
        fusion_loss_total = lamda_fusion * ssim_loss_combined +  (1-lamda_fusion) * l2_loss_combined
        
        
        optimizer.zero_grad()
        fusion_loss_total.backward()
        optimizer.step()
        #scheduler.step()
            
        #store the training loss value at each epoch
        loss_train.append(fusion_loss_total.item())
        ssim_t1ce_train.append(ssim_loss_t1ce.item())
        ssim_flair_train.append(ssim_loss_flair.item())
        loss_ssim_train_t1ce.append(ssim_loss_t1ce.item())
        loss_ssim_train_flair.append(ssim_loss_flair.item())
        loss_l2_train_t1ce.append(l2_loss_t1ce.item())
        loss_l2_train_flair.append(l2_loss_flair.item())  
        count = count + 1          
            
            
    av_train_loss = np.average(loss_train)
    ep_train_loss.append(av_train_loss)
    
    av_ssim_t1ce_train = np.average(ssim_t1ce_train)
    ep_ssim_t1ce_train.append(av_ssim_t1ce_train)
    av_ssim_flair_train = np.average(ssim_flair_train)
    ep_ssim_flair_train.append(av_ssim_flair_train)

    av_ssim_train_loss_t1ce = np.average(loss_ssim_train_t1ce)
    ep_ssim_train_loss_t1ce.append(av_ssim_train_loss_t1ce)
    av_l2_train_loss_t1ce = np.average(loss_l2_train_t1ce)
    ep_l2_train_loss_t1ce.append(av_l2_train_loss_t1ce)
    
    av_ssim_train_loss_flair = np.average(loss_ssim_train_flair)
    ep_ssim_train_loss_flair.append(av_ssim_train_loss_flair)
    av_l2_train_loss_flair = np.average(loss_l2_train_flair)
    ep_l2_train_loss_flair.append(av_l2_train_loss_flair)        
        
        
    #Validation of the model.
    model.eval()
    
    batch_idxs_val = total_val_images // batch_size
    with torch.no_grad():
        for idx in range(0, batch_idxs_val):
            input_val  = val_data_tensor[idx*batch_size : (idx+1)*batch_size,:,:,:].to(device)
                
            
            weight_map_val = model(input_val)
            fused_val = weight_map_val*input_val[:,0:1,:,:] + (1-weight_map_val)*input_val[:,1:2,:,:]                
                
            if epoch % 1 == 0:
                path_im = f"/home/h3/issr292b/image_fusion/val_samples/en-de/ssim_0.1/"
                path_weight_val = f"/home/h3/issr292b/image_fusion/val_samples_fused/en-de/ssim_0.1/"
                os.makedirs(path_im, exist_ok=True)
                os.makedirs(path_weight_val, exist_ok=True)
                images_concat = torchvision.utils.make_grid(weight_map_val, nrow=int(weight_map_val.shape[0] ** 0.5), padding=2, pad_value=255)
                weight_val = torchvision.utils.make_grid(weight_map_val, nrow=int(fused_val.shape[0] ** 0.5), padding=2, pad_value=255)
                torchvision.utils.save_image(weight_val, '/home/h3/issr292b/image_fusion/val_samples/en-de/ssim_0.1/epoch_{}.png'.format(epoch))
                torchvision.utils.save_image(images_concat, '/home/h3/issr292b/image_fusion/val_samples_fused/en-de/ssim_0.1/epoch_{}.png'.format(epoch))

                
        
            #SSIM loss for the fusion training
            ssim_module = SSIM(data_range=255, size_average=True, channel=3)
            ssim_loss_t1ce_val  = 1 - ssim_module(weight_map, input_[:,0:3,:,:])
            ssim_loss_flair_val = 1 - ssim_module(weight_map, input_[:,3:6,:,:])
            #club the T1ce and flair ssim losses
            ssim_loss_combined_val = lamda_ssim * ssim_loss_t1ce_val + (1-lamda_ssim) * ssim_loss_flair_val
        
            #l2 loss for the fusion training
            l2_loss_t1ce_val  = l2_loss(weight_map, input_[:,0:3,:,:])
            l2_loss_flair_val = l2_loss(weight_map, input_[:,3:6,:,:])   
             
            #club the T1ce and flair l2 losses
            l2_loss_combined_val = lamda_l2 * l2_loss_t1ce_val + (1-lamda_l2) * l2_loss_flair_val     
        
            #combine ssim_l2_loss
            fusion_loss_total_val = lamda_fusion * ssim_loss_combined_val +  (1-lamda_fusion) * l2_loss_combined_val
                
            loss_val.append(fusion_loss_total_val.item())
            ssim_t1ce_val.append(ssim_loss_t1ce_val.item())
            ssim_flair_val.append(ssim_loss_flair_val.item())
            loss_ssim_val_t1ce.append(ssim_loss_t1ce_val.item())
            loss_ssim_val_flair.append(ssim_loss_flair_val.item())
            loss_l2_val_t1ce.append(l2_loss_t1ce_val.item())
            loss_l2_val_flair.append(l2_loss_flair_val.item())    
                
                
        av_val_loss = np.average(loss_val)
        ep_val_loss.append(av_val_loss)

        av_ssim_t1ce_val = np.average(ssim_t1ce_val)
        ep_ssim_t1ce_val.append(av_ssim_t1ce_val)
        av_ssim_flair_val = np.average(ssim_flair_val)
        ep_ssim_flair_val.append(av_ssim_flair_val)

        av_ssim_val_loss_t1ce = np.average(loss_ssim_val_t1ce)
        ep_ssim_val_loss_t1ce.append(av_ssim_val_loss_t1ce)
        av_l2_val_loss_t1ce = np.average(loss_l2_val_t1ce)
        ep_l2_val_loss_t1ce.append(av_l2_val_loss_t1ce)
        
        av_ssim_val_loss_flair = np.average(loss_ssim_val_flair)
        ep_ssim_val_loss_flair.append(av_ssim_val_loss_flair)
        av_l2_val_loss_flair = np.average(loss_l2_val_flair)
        ep_l2_val_loss_flair.append(av_l2_val_loss_flair) 
        
        
    if epoch == 100:    
        path = f"/home/h3/issr292b/image_fusion/epoch/en-de/"
        os.makedirs(path, exist_ok=True)
        # Save optimization status. We should save the objective value because the process may be
        # killed between saving the last model and recording the objective value to the storage.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                #"scheduler_state_dict": scheduler.state_dict(),
                "ssim loss train t1ce": ssim_t1ce_train,
                "ssim loss train flair": ssim_flair_train,
                "ssim loss val t1ce": ssim_t1ce_val,
                "ssim loss val flair": ssim_flair_val,
                "training_loss_total": ep_train_loss,
                "validation_loss_total": ep_val_loss,
                "avg. training_loss_ssim_t1ce": ep_ssim_train_loss_t1ce,
                "validation_loss_ssim_t1ce": ep_ssim_val_loss_t1ce,
                "avg. training_loss_ssim_flair": ep_ssim_train_loss_flair,                     
                "validation_loss_ssim_flair": ep_ssim_val_loss_flair,
                "training_loss_l2_t1ce": ep_l2_train_loss_t1ce,
                "validation_loss_l2_t1ce": ep_l2_val_loss_t1ce,
                "training_loss_l2_flair": ep_l2_train_loss_flair,    
                "validation_loss_l2_flair": ep_l2_val_loss_flair
            },
            os.path.join(path, "model_0.1.pt"))


