from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3) 
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)

plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()



