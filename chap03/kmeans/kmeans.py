from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# generate synthetic two-dimensional data 
X, y = make_blobs(random_state=1)

# build the clustering model 
kmeans = KMeans(n_clusters=3) 
kmeans.fit(X)

print("Cluster memberships:\n{}".format(kmeans.labels_))
print(kmeans.predict(X))

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)

plt.show()