import cv2	
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mua_muong_teT52019_8_1.png')
edges = cv2.Canny(img, 50,150)

lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,
	threshold = 100,minLineLength = 100,maxLineGap = 50)

N = lines.shape[0]
for i in range(N):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]    
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
cv2.inRange(img, np.array([50, 150, 150]), np.array([125, 225, 255]))

cv2.imshow('anhhh',img)
cv2.imshow('image',edges)	


cv2.waitKey(0)
cv2.destroyAllWindows()