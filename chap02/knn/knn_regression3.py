import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()