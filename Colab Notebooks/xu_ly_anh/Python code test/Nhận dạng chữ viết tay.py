import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('anh so.png',0)

cv2.imshow('check',img)

cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
print(cells[0][0])

cv2.imwrite('so.jpg',cells[0][0])

cv2.waitKey(0)
cv2.destroyAllWindows()