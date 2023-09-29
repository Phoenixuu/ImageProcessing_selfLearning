import cv2

img = cv2.imread('mua_muong_teT52019_8_1.png',0)

cv2.imshow('Gray',img)

ret, bw = cv2.threshold(img,127,120,cv2.THRESH_BINARY)

cv2.imshow("Binary",bw)

cv2.waitKey(0)

cv2.destroyAllWindows()