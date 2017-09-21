import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# knn
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()