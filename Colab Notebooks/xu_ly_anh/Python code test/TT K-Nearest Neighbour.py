import cv2
import numpy as np 
import matplotlib.pyplot as plt

a = np.random.randint(0, 100, (25, 2)).astype(np.float32)
b = np.random.randint(0, 2, (25, 1)).astype(np.float32)
red = a[b.ravel()==1]
blue =a[b.ravel()==0]
newMember = np.random.randint(0, 100, (1, 2)).astype(np.float32)

# print(a)
# print(b)
# print(red)

# print(red[:,0])

plt.scatter(blue[:,0], blue[:,1], 100, 'b','^')
plt.scatter(red[:,0], red[:,1], 100, 'r','s')
plt.scatter(newMember[:,0], newMember[:,1], 100, 'g','o')

knn = cv2.ml.KNearest_create()
knn.train(a, 0, b)
results = knn.findNearest(newMember, 3)

plt.show()
print(results) 