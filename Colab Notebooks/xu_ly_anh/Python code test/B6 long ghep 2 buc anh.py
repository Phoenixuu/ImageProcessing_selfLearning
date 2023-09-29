import cv2;
import numpy as np

img1 = cv2.imread('nang.jpg',1)
img2 = cv2.imread('hpny.jpg',1)

img1 = img1[100:600, 100:800]
img2 = img2[100:600, 100:800]
img3 = cv2.add(img1, img2)
# cv2.imshow('image1',img1)
# cv2.imshow('image2',img2)
cv2.imshow('image3',img3)
cv2.waitKey(0)