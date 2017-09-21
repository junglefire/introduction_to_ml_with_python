from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN() 
clusters = dbscan.fit_predict(X) 
print("Cluster memberships:\n{}".format(clusters))

mglearn.plots.plot_dbscan()
plt.show()



