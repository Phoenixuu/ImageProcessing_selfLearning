import cv2
import numpy as np

image = cv2.imread('mua_muong_teT52019_8_1.png')
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#phat hien canh canny co thanh mau xam 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canimg = cv2.Canny(gray, 50,200)
# #phuong phap houghlines, bo tich luy tinh bang radian, do phan giai khoang cach  
lines = cv2.HoughLines(canimg, 1,np.pi/180, 120, np.array([]))
#Vecto phan tu rho va theta 
for line in lines:
#rho:khoang cach toa do khong tren cung ben trai
#theta:goc quay cua duong thang tinh bang radian
	rho,theta = line[0]
	a = np.cos(theta)
	b = np.sin(theta)

	x0 = a*rho
	y0 = b*rho

	#x1 = r * cos(theta) - 1000 * sin(theta)
	x1 = int(x0+1000*(-b))
	#y2 = r * sin(theta) + 1000 * cos(theta) 
	y1 = int(y0 + 1000*(a))
	#x2 = r * cos(theta) + 1000 * sin(theta)
	x2 = int(x0-1000*(-b))
	#y2 = r * sin(theta) - 1000 * coss\(theta)
	y2 = int(y0 - 1000*(a))

	cv2.line(image,(x1,y1),(x2,y2), (0,0,255), 2)

def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)

    for i in range(0,len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt


    homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

    return homography


canimg = cv2.resize(canimg,dsize = (int(canimg.shape[1]* 0.3), int(canimg.shape[0]* 0.3)))
purple_min = np.array([110, 0, 130], np.uint8)
purple_max = np.array([170, 150, 250], np.uint8)
img_threshed = cv2.inRange(img_hsv, purple_min, purple_max)


kernel = np.ones((3, 3), np.uint8)
img_closing = cv2.morphologyEx(img_threshed, cv2.MORPH_CLOSE, kernel)


cv2.imshow('Lines Detected',image)
cv2.imshow('Canny Dectection', canimg)
cv2.imshow('result',img_closing)


cv2.imwrite('AMT.jpg',canimg)
cv2.imwrite('AMTline.jpg',img_closing)
cv2.waitKey(0)
cv2.destroyAllWindows(0)

ulike()