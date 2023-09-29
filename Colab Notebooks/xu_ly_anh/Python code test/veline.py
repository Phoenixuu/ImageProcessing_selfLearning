import cv2 
import numpy as np

img = cv2.imread('mua_muong_teT52019_8_1.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#canny = cv2.Canny(imgGray, 50, 200)
canny = cv2.inRange(img, np.array([50, 150, 150]), np.array([125, 225, 255]))
lines = cv2.HoughLinesP(canny,rho = 1,theta = 1*np.pi/180,threshold = 100,
	minLineLength = 100,maxLineGap = 100)
N = lines.shape[0]
for i in range(N):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]    
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
cv2.imwrite('det_line.jpg', canny)
cv2.imwrite('line_det_canny.jpg', img) 

cv2.imshow('anhmau', canny)
cv2.imshow('anhkomau', img)

cv2.waitKey(0)
cv2.DestroyAllWindow()