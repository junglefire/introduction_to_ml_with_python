from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier 
# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

cancer = load_breast_cancer() 
print("cancer.keys(): \n{}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))


