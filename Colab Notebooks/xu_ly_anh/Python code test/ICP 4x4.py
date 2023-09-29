import cv2
import numpy as np
import imutils
import sys
from imutils import contours

import cv2 as cv
import argparse


img = cv2.imread('anh crop1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,255,1)
cv2.imshow('anh goc',img)

edges = cv2.Canny(img, 50,150)
cv2.imshow('edges',edges)
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('Corner',img)
    
x = len(corners)
print(x)

a = [50][50]

thresh = 10

for i in range(0,x):
    t1,t2 = corner[i]
    for j in range(0,x):
        s1,s2 = corner[j]        
        d1 = abs(s1-t1)
        if d1 < thresh:
            a[i][j] = 1
        else:
            a[i][j] = 0

cv2.imshow("matran",a)





#1. điểm corner
#2. sinh ra ma trận khoảng cách
#3. duyệt trên từng hàng





# # Lọc nhiễu
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 1000:
#         cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

# # Tìm các trục tung và trục hoành
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=4)
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

# # Sắp xếp các dòng line
# invert = 255 - thresh
# cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

# mua_rows = []
# row = []
# for (i, c) in enumerate(cnts, 1):
#     area = cv2.contourArea(c)
#     if area < 50000:
#         row.append(c)
#         if i % 4 == 0:  
#             (cnts, _) = contours.sort_contours(row, method="left-to-right")
#             mua_rows.append(cnts)
#             row = []

# for row in mua_rows:
#     for c in row:
#         mask = np.zeros(img.shape, dtype=np.uint8)
#         cv2.drawContours(mask, [c], -1, (255,255,255), -1)
#         result = cv2.bitwise_and(img, mask)
#         result[mask==0] = 255
#         cv2.imshow('result', result)
#         cv2.waitKey(175)

# cv2.imshow('thresh', thresh)
# cv2.imshow('invert', invert)

# cv2.waitKey(0)





# parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='anh crop1.png')
# args = parser.parse_args()
# src = cv.imread(cv.samples.findFile(args.input))
# cv.IMREAD_GRAYSCALE
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
# #-- Step 1: Detect the keypoints using SURF Detector
# minHessian = 400
# detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
# keypoints = detector.detect(src)
# #-- Draw keypoints
# img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
# cv.drawKeypoints(src, keypoints, img_keypoints)
# #-- Show detected (drawn) keypoints
# cv.imshow('SURF Keypoints', img_keypoints)
# cv.waitKey()



# parser = argparse.ArgumentParser(description='Code for Affine Transformations tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='anh crop1.png')
# args = parser.parse_args()
# src = cv.imread(cv.samples.findFile(args.input))
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
# srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
# dstTri = np.array( [[0, src.shape[1]*0.33], [src.shape[1]*0.85, src.shape[0]*0.25], [src.shape[1]*0.15, src.shape[0]*0.7]] ).astype(np.float32)
# warp_mat = cv.getAffineTransform(srcTri, dstTri)
# warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
# # Rotating the image after Warp
# center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
# angle = -50
# scale = 0.6
# rot_mat = cv.getRotationMatrix2D( center, angle, scale )
# warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
# cv.imshow('Source image', src)
# cv.imshow('Warp', warp_dst)
# cv.imshow('Warp + Rotate', warp_rotate_dst)
# cv.waitKey()


cv2.destroyAllWindows()



