import cv2
import numpy as np

img = cv2.imread("anh crop1.png", 1)
# print(img.size)
subimg = img[3000:4000, 5000:6600]
subimg = subimg[:,:,2]
cv2.imshow('cat',subimg)
cv2.waitKey()