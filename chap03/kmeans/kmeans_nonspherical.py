from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# generate some random cluster data 
X, y = make_blobs(random_state=170, n_samples=600) 
rng = np.random.RandomState(74)

# transform the data to be stretched 
transformation = rng.normal(size=(2, 2)) 
X = np.dot(X, transformation)

# cluster the data into three clusters 
kmeans = KMeans(n_clusters=3) 
kmeans.fit(X) 
y_pred = kmeans.predict(X)

# plot the cluster assignments and cluster centers 
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3) 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c=[0, 1, 2], s=100, linewidth=2, cmap=mglearn.cm3) 
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()