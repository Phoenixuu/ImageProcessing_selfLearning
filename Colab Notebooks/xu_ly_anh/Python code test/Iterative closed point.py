#http://www.onerussian.com/classes/cis780/icp-slides.pdf

import numpy as np  
import matplotlib.pyplot as plt  


# meshgrid
# https://www.javatpoint.com/numpy-meshgrid
import numpy as np  
import matplotlib.pyplot as plt  
a = np.arange(-10, 10, 0.1)  
b = np.arange(-10, 10, 0.1)  
xa, xb = np.meshgrid(a, b, sparse=True)  
z = np.sin(xa**2 + xb**2) / (xa**2 + xb**2)  
h = plt.contourf(a,b,z)  
plt.show()  

