import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles 

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
plt.plot(X[:,0], X[:,1], 'o')
plt.xlabel("X")
plt.ylabel("y")
plt.show()