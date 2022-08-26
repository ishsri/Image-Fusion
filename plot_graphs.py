
from cProfile import label
import torch
import numpy as np
import matplotlib.pyplot as plt

checkpoint_fusion = torch.load(r'/home/h3/issr292b/image_fusion/epoch/deep_fuse/model_0.0.pt')

loss_ssim_train_t1ce = checkpoint_fusion['training_loss_ssim_t1ce']
loss_ssim_train_flair = checkpoint_fusion['training_loss_ssim_flair']
loss_l2_train_t1ce = checkpoint_fusion['training_loss_l2_t1ce']
loss_l2_train_flair = checkpoint_fusion['training_loss_l2_flair']
loss_ssim_val_t1ce = checkpoint_fusion['validation_loss_ssim_t1ce']
loss_ssim_val_flair = checkpoint_fusion['validation_loss_ssim_flair']
loss_l2_val_t1ce = checkpoint_fusion['validation_loss_l2_t1ce']
loss_l2_val_flair = checkpoint_fusion['validation_loss_l2_flair']

loss_ssim_train_t1ce = np.asarray(loss_ssim_train_t1ce).flatten()
loss_ssim_train_flair = np.asarray(loss_ssim_train_flair).flatten()
loss_l2_train_t1ce = np.asarray(loss_l2_train_t1ce).flatten()
loss_l2_train_flair = np.asarray(loss_l2_train_flair).flatten()
loss_ssim_val_t1ce = np.asarray(loss_ssim_val_t1ce).flatten()
loss_ssim_val_flair = np.asarray(loss_ssim_val_flair).flatten()
loss_l2_val_t1ce = np.asarray(loss_l2_val_t1ce).flatten()
loss_l2_val_flair = np.asarray(loss_l2_val_flair).flatten()

#rand_var = loss_l2_train_flair - loss_l2_train_t1ce

plt.plot(loss_ssim_train_t1ce,'b', label = 'ssim train_t1ce')
plt.plot(loss_ssim_val_t1ce,'g', label = 'ssim val_t1ce')
#plt.plot(loss_l2_train_t1ce,'r', label = 'l2_train_t1ce')
#plt.plot(loss_l2_val_t1ce,'c', label = 'l2_val_t1ce')
plt.title('MRI t1ce _ SSIM _ 0.49')
plt.legend()
plt.savefig('image_2.1.png', dpi=200)
plt.close()

# plt.plot(loss_ssim_train_flair,'b', label = 'ssim train_flair')
# plt.plot(loss_ssim_val_flair,'g', label = 'ssim val_flair')
# plt.plot(loss_l2_train_flair,'r', label = 'l2_train_flair')
# plt.plot(loss_l2_val_flair,'c', label = 'l2_val_flair')
# plt.title('MRI flair _ SSIM _ 0.49')
# plt.legend()
# plt.savefig('image_2.png', dpi=200)
# plt.close()


