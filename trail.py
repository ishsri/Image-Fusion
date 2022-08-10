from statistics import mode
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import imageio
import torch
import matplotlib.pyplot as plt
image = imageio.imread(r"C:\Users\ishan\Desktop\Image Fusion\Fusion_R1\2.png") 

model = torch.load(r'C:\Users\ishan\Desktop\Image Fusion\Fusion_R1\model_0.49.pt')
model_state = model["model_state_dict"]
# df = pd.DataFrame.from_dict(model_state)

print(model_state["conv13.0.weight"])
final_weight = model_state["conv13.0.weight"]
final_weight = final_weight.cpu().numpy()
print(final_weight.shape)

imgplot = plt.imshow(final_weight)