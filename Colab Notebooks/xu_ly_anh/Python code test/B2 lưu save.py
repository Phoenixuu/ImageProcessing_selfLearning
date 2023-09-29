import numpy as np
import cv2

img = cv2.imread("me.JPG",1)
cv2.line(img, (-1200, 1200),(1000,1000), (255,22,234), 30)
cv2.imwrite('dave.jpg', img)
cv2.imshow('ve len anh', img)


cv2.waitKey(0)
cv2.destroyAllWindows()