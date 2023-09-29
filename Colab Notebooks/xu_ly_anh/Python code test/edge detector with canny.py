import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('am_muong_te_T72007_10_2.png',1)
edges = cv.Canny(img,100,200)

lines = cv.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,
	threshold = 100,minLineLength = 100,maxLineGap = 50)

cv.inRange(img, np.array([50, 150, 150]), np.array([125, 225, 255]))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

cv.imwrite('xulyanh.jpg',edges)
cv.imwrite('anhmauu.jpg',img)

cv.waitKey(0)
cv.destroyAllWindows()
plt.show()

