from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn



# these two labelings of points correspond to the same clustering
clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]

# accuracy is zero, as none of the labels are the same 
print("Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2))) 
# adjusted rand score is 1, as the clustering is exactly the same 
print("ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))



