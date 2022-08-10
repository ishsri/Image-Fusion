import cv2

image = cv2.imread(r'C:/Users/ishan/Desktop/Image Fusion/Fusion_R1/train_samples/count_3000.png')

cv2.imshow('Original', image)
cv2.waitKey(0)

print(image.shape)