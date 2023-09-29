import numpy as np
import cv2
from matplotlib import pyplot as plt

def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0,len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt


    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography



# def drawMatches(img1, kq1, img2, kq2, matches):
# 	rows1 = img1.shape[0]
# 	cols1 = img1.shape[1]
# 	rows2 = img2.shape[0]
# 	cols2 = img2.shape[1]

# 	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype = 'unit8')

# 	#Place the first image to the left
# 	out[:rows2,cols1:] = np.dstack([img1,img1,img1])

# 	out[:rows2,cols1:] = np.dstack([img2,img2,img2]) 

# 	for mat in matches:

# 		img1_idx = mat.queryIdx
# 		img2_idx = mat.trainIdx

# 		(x1,y1) = kq1[img1_idx].pt
# 		(x2.y2) = kq2[img2_idx].pt

# 		cv2.circle(out, (int(x1),int(y1)), 4, (255,0,0),1)
# 		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255,0,0),1)

# 		cv2.line(out,(int(x1),int(y1)),(int(x2)+cols1,int(y2)),(255,0,0),1)

# 	cv2.imshow('Matched Features',out)
# 	cv2.waitkey(0)
# 	cv2.destroyWindow('Matched Features')

# 	return out

# img1 = cv2.imread('sudoku.jpg')
# img2 = cv2.imread('mua_muong_teT52019_8_1.png')

# name = cv2.COLOR_YUV2BGRA_YV12
# # print name
# gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT()
# kq1,des1 = sift.detectAndCompute(gray1, None)
# kq2,des2 = sift.detectAndCompute(gray2, None)
# bf = cv2.BFMacher()
# matches = bf.match(des1,des2)
# matches = sorted(matches,key=lambda x:x.distance)
# img3 = drawMatches((gray1), kq1, gray2, kq2, matches[:100])
# plt.imshow(img3,plt.show())

# print(matches)