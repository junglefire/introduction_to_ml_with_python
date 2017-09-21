from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# generate synthetic two-dimensional data 
X, y = make_blobs(random_state=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# using two cluster centers: 
kmeans = KMeans(n_clusters=2) 
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

# using five cluster centers: 
kmeans = KMeans(n_clusters=5) 
kmeans.fit(X) 
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

plt.show()