from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale the data to zero mean and unit variance 
scaler = StandardScaler() 
scaler.fit(X) 
X_scaled = scaler.transform(X)

dbscan = DBSCAN() 
clusters = dbscan.fit_predict(X_scaled) 

# plot the cluster assignments 
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60) 
plt.xlabel("Feature 0") 
plt.ylabel("Feature 1")
plt.show()



