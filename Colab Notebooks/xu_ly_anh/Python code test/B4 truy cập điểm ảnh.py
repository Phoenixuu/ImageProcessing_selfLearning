import cv2
import numpy as np

img = cv2.imread("anh crop1.png",1)
# cv2.imshow("image",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

px = img[200][200]
print(px)

for i in range(200):
	for j in range(200):
		if img[i,j,0] > 5:
			img[i,j]=1
px = img[200][200]

cv2.imwrite('sua2.jpg',img) 