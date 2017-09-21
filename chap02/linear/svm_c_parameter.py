from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

mglearn.plots.plot_linear_svc_regularization()
plt.show()