from sklearn.datasets import load_boston

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

boston = load_boston()
print("Data shape: {}".format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston() 
print("X.shape: {}".format(X.shape))

