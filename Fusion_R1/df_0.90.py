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


hf = h5py.File("/projects/p084/p_discoret/Brats2018_training_data_sep_channels_train_val_mix.h5", 'r')
train_data = hf['data'][()]     #`data` is now an ndarray
hf.close()

hf = h5py.File("/projects/p084/p_discoret/Brats2018_training_data_sep_channels_train_val_mix.h5", 'r')
train_labels = hf['label'][()]     #`data` is now an ndarray
hf.close()

hf = h5py.File("/projects/p084/p_discoret/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
val_data = hf['data'][()]     #`data` is now an ndarray
hf.close()

hf = h5py.File("/projects/p084/p_discoret/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
val_labels = hf['label'][()]     #`data` is now an ndarray
hf.close()



for i in range(len(train_data)):
    for j in range(4):
        train_data[i,j,:,:] =(train_data[i,j,:,:] - np.min(train_data[i,j,:,:])) / (np.max(train_data[i,j,:,:]) - np.min(train_data[i,j,:,:]))
        
        
for i in range(len(val_data)):
    for j in range(4):
        val_data[i,j,:,:] =(val_data[i,j,:,:] - np.min(val_data[i,j,:,:])) / (np.max(val_data[i,j,:,:]) - np.min(val_data[i,j,:,:]))
		
		
		
		
#do it for the training data
train_t1_tensor = torch.from_numpy(train_data[:,0,:,:]).float()
train_t1_tensor = train_t1_tensor[:, None, :, :]

train_t2_tensor = torch.from_numpy(train_data[:,1,:,:]).float()
train_t2_tensor = train_t2_tensor[:, None, :, :]

train_t1ce_tensor = torch.from_numpy(train_data[:,2,:,:]).float()
train_t1ce_tensor = train_t1_tensor[:, None, :, :]

train_flair_tensor = torch.from_numpy(train_data[:,3,:,:]).float()
train_flair_tensor = train_flair_tensor[:, None, :, :]

train_label_tensor = torch.from_numpy(train_labels).float()


#do it for the validation data
val_t1_tensor = torch.from_numpy(val_data[:,0,:,:]).float()
val_t1_tensor = val_t1_tensor[:, None, :, :]

val_t2_tensor = torch.from_numpy(val_data[:,1,:,:]).float()
val_t2_tensor = val_t2_tensor[:, None, :, :]

val_t1ce_tensor = torch.from_numpy(val_data[:,2,:,:]).float()
val_t1ce_tensor = val_t1ce_tensor[:, None, :, :]

val_flair_tensor = torch.from_numpy(val_data[:,3,:,:]).float()
val_flair_tensor = val_flair_tensor[:, None, :, :]

val_label_tensor = torch.from_numpy(val_labels).float()


#do it for the training data
train_data_tensor = torch.from_numpy(train_data).float()
train_label_tensor = torch.from_numpy(train_labels).float()

val_data_tensor = torch.from_numpy(val_data).float()
val_label_tensor = torch.from_numpy(val_labels).float()



total_train_images = 8500
total_val_images = 1153
EPOCHS = 101
batch_size = 64
gpu_ids = [0,1]


# -------------------------------------------------------------------------------------------------------
#   Define DeepFuse Network
# -------------------------------------------------------------------------------------------------------
class ConvLayer_DeepFuse(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, kernel_size = 5, last = nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class FusionLayer(nn.Module):
    def forward(self, x, y):
        return x + y


class DeepFuse(nn.Module):
    def __init__(self):
        super(DeepFuse, self).__init__()
        self.layer1 = ConvLayer_DeepFuse(1, 16, 5, last = nn.LeakyReLU)
        self.layer2 = ConvLayer_DeepFuse(16, 32, 7)
        self.layer3 = FusionLayer()
        self.layer4 = ConvLayer_DeepFuse(32, 32, 7, last = nn.LeakyReLU)
        self.layer5 = ConvLayer_DeepFuse(32, 16, 5, last = nn.LeakyReLU)
        self.layer6 = ConvLayer_DeepFuse(16, 1, 5, last = nn.Tanh)

    def setInput(self, y_1, y_2):
        self.y_1 = y_1
        self.y_2 = y_2

    def forward(self):
        c11 = self.layer1(self.y_1)
        c12 = self.layer1(self.y_2)
        c21 = self.layer2(c11)
        c22 = self.layer2(c12)
        f_m = self.layer3(c21, c22)
        c3  = self.layer4(f_m)
        c4  = self.layer5(c3)
        c5  = self.layer6(c4)
        return c5




model = DeepFuse().to(device)
model = model.float()
if device == 'cuda':
    net = torch.nn.DataParallel(model, gpu_ids)
    cudnn.benchmark = True
        
#define the optimizers and loss functions 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)   # optimize all cnn parameters
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 0.95 ** epoch)

l2_loss   = nn.MSELoss() #MSEloss  
    
lamda_ssim = 0.90
lamda_l2 = 0.49
lamda_fusion = 0.99
    
loss_train = []
loss_ssim_train_t1ce = []
loss_ssim_train_flair = []
loss_l2_train_t1ce = []
loss_l2_train_flair = []
        
loss_val = []
loss_ssim_val_t1ce = []
loss_ssim_val_flair = []
loss_l2_val_t1ce = []
loss_l2_val_flair = []
    
    
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
        input_x  = train_data_tensor[idx*batch_size : (idx+1)*batch_size,2:3,:,:].to(device)
        input_y  = train_data_tensor[idx*batch_size : (idx+1)*batch_size,3:,:,:].to(device)
        weight_map = model(input_x, input_y)
        fused = weight_map*input_x + (1-weight_map)*input_y

        
        path_im = f"/home/h3/issr292b/image_fusion/train_samples/deep_fuse/ssim_0.90/"
        path_fused = f"/home/h3/issr292b/image_fusion/train_samples_fused/deep_fuse/ssim_0.90/"
        os.makedirs(path_im, exist_ok=True)
        os.makedirs(path_fused, exist_ok=True)
        images_concat = torchvision.utils.make_grid(weight_map, nrow=int(weight_map.shape[0] ** 0.5), padding=2, pad_value=255)
        fused_save = torchvision.utils.make_grid(fused, nrow = int(fused.shape[0] ** 0.5), padding=2, pad_value=255)
        if count % 10 == 0:
            torchvision.utils.save_image(images_concat, '/home/h3/issr292b/image_fusion/train_samples/deep_fuse/ssim_0.90/count_{}.png'.format(count))
            torchvision.utils.save_image(fused_save, '/home/h3/issr292b/image_fusion/train_samples_fused/deep_fuse/ssim_0.90/count_{}.png'.format(count))
          
        #SSIM loss for the fusion training
        ssim_loss_t1ce  = 1 - ssim(fused, input_x,data_range=1)
        ssim_loss_flair = 1 - ssim(fused, input_y,data_range=1)
        #club the T1ce and flair ssim losses
        ssim_loss_combined = lamda_ssim * ssim_loss_t1ce + (1-lamda_ssim) * ssim_loss_flair
        
        #l2 loss for the fusion training
        l2_loss_t1ce  = l2_loss(fused, input_x)
        l2_loss_flair = l2_loss(fused, input_y)        
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
        loss_ssim_train_t1ce.append(ssim_loss_t1ce.item())
        loss_ssim_train_flair.append(ssim_loss_flair.item())
        loss_l2_train_t1ce.append(l2_loss_t1ce.item())
        loss_l2_train_flair.append(l2_loss_flair.item())  
        count = count + 1          
            
            
    av_train_loss = np.average(loss_train)
    ep_train_loss.append(av_train_loss)
    
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
            input_val_x  = val_data_tensor[idx*batch_size : (idx+1)*batch_size,2:3,:,:].to(device)
            input_val_y  = val_data_tensor[idx*batch_size : (idx+1)*batch_size,3:,:,:].to(device)
                
            
            weight_map_val = model(input_val_x, input_val_y)
            fused_val = weight_map_val*input_val_x + (1-weight_map_val)*input_val_y                
                
            if epoch % 1 == 0:
                path_im = f"/home/h3/issr292b/image_fusion/val_samples/deep_fuse/ssim_0.90/"
                path_weight_val = f"/home/h3/issr292b/image_fusion/val_samples_fused/deep_fuse/ssim_0.90/"
                os.makedirs(path_im, exist_ok=True)
                os.makedirs(path_weight_val, exist_ok=True)
                images_concat = torchvision.utils.make_grid(fused_val, nrow=int(fused_val.shape[0] ** 0.5), padding=2, pad_value=255)
                weight_val = torchvision.utils.make_grid(weight_map_val, nrow=int(fused_val.shape[0] ** 0.5), padding=2, pad_value=255)
                torchvision.utils.save_image(weight_val, '/home/h3/issr292b/image_fusion/val_samples/deep_fuse/ssim_0.90/epoch_{}.png'.format(epoch))
                torchvision.utils.save_image(images_concat, '/home/h3/issr292b/image_fusion/val_samples_fused/deep_fuse/ssim_0.90/epoch_{}.png'.format(epoch))

                
            ssim_loss_t1ce_val  = 1 - ssim(fused_val, input_val_x,data_range=1)
            ssim_loss_flair_val = 1 - ssim(fused_val, input_val_y,data_range=1)
            #club the T1ce and flair ssim losses
            ssim_loss_combined_val = lamda_ssim * ssim_loss_t1ce_val + (1-lamda_ssim) * ssim_loss_flair_val
            
            #l2 loss for the fusion training
            l2_loss_t1ce_val  = l2_loss(fused_val, input_val_x)
            l2_loss_flair_val = l2_loss(fused_val, input_val_y)   
            #club the T1ce and flair l2 losses
            l2_loss_combined_val = lamda_l2 * l2_loss_t1ce_val + (1-lamda_l2) * l2_loss_flair_val     
        
            #combine ssim_l2_loss
            fusion_loss_total_val = lamda_fusion * ssim_loss_combined_val +  (1-lamda_fusion) * l2_loss_combined_val
                
            loss_val.append(fusion_loss_total_val.item())
            loss_ssim_val_t1ce.append(ssim_loss_t1ce_val.item())
            loss_ssim_val_flair.append(ssim_loss_flair_val.item())
            loss_l2_val_t1ce.append(l2_loss_t1ce_val.item())
            loss_l2_val_flair.append(l2_loss_flair_val.item())    
                
                
        av_val_loss = np.average(loss_val)
        ep_val_loss.append(av_val_loss)
        
        av_ssim_val_loss_t1ce = np.average(loss_ssim_val_t1ce)
        ep_ssim_val_loss_t1ce.append(av_ssim_val_loss_t1ce)
        av_l2_val_loss_t1ce = np.average(loss_l2_val_t1ce)
        ep_l2_val_loss_t1ce.append(av_l2_val_loss_t1ce)
        
        av_ssim_val_loss_flair = np.average(loss_ssim_val_flair)
        ep_ssim_val_loss_flair.append(av_ssim_val_loss_flair)
        av_l2_val_loss_flair = np.average(loss_l2_val_flair)
        ep_l2_val_loss_flair.append(av_l2_val_loss_flair) 
        
        
    if epoch == 100:    
        path = f"/home/h3/issr292b/image_fusion/epoch/deep_fuse/"
        os.makedirs(path, exist_ok=True)
        # Save optimization status. We should save the objective value because the process may be
        # killed between saving the last model and recording the objective value to the storage.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                #"scheduler_state_dict": scheduler.state_dict(),
                "training_loss_total": ep_train_loss,
                "validation_loss_total": ep_val_loss,
                "training_loss_ssim_t1ce": ep_ssim_train_loss_t1ce,
                "validation_loss_ssim_t1ce": ep_ssim_val_loss_t1ce,
                "training_loss_ssim_flair": ep_ssim_train_loss_flair,                     
                "validation_loss_ssim_flair": ep_ssim_val_loss_flair,
                "training_loss_l2_t1ce": ep_l2_train_loss_t1ce,
                "validation_loss_l2_t1ce": ep_l2_val_loss_t1ce,
                "training_loss_l2_flair": ep_l2_train_loss_flair,    
                "validation_loss_l2_flair": ep_l2_val_loss_flair
            },
            os.path.join(path, "model_0.90.pt"))


